"""Local financial-news sentiment backfill utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import bindparam, select, update

from quant_trading_system.database.connection import DatabaseManager, get_db_manager
from quant_trading_system.database.models import NewsArticle

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _combine_news_text(headline: str | None, summary: str | None) -> str:
    parts = [str(part).strip() for part in (headline, summary) if str(part or "").strip()]
    return "\n\n".join(parts)


def _label_scores_to_sentiment(label_scores: list[dict[str, Any]]) -> float:
    mapping = {
        str(item.get("label") or "").strip().lower(): float(item.get("score") or 0.0)
        for item in label_scores
    }
    return float(mapping.get("positive", 0.0) - mapping.get("negative", 0.0))


@dataclass
class LocalNewsSentimentConfig:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 32
    source: str | None = "alpaca"
    only_null_sentiment: bool = True
    limit: int | None = None
    backend: str = "auto"


class LocalNewsSentimentBackfiller:
    """Backfill article-level sentiment scores using a local transformer model."""

    def __init__(
        self,
        config: LocalNewsSentimentConfig,
        *,
        db_manager: DatabaseManager | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.db_manager = db_manager or get_db_manager()
        self.logger = logger_ or logger
        self._pipeline = None
        self._transformer_disabled = False
        self._vader_analyzer = None

    def _build_transformer_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise RuntimeError("Transformer sentiment backend is unavailable.") from exc

        device = 0 if torch.cuda.is_available() else -1
        self.logger.info(
            "Loading local news sentiment model %s on %s",
            self.config.model_name,
            "cuda" if device >= 0 else "cpu",
        )
        self._pipeline = pipeline(
            "text-classification",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            device=device,
            top_k=None,
            truncation=True,
        )
        return self._pipeline

    def _build_vader_analyzer(self):
        if self._vader_analyzer is not None:
            return self._vader_analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except Exception as exc:  # pragma: no cover - dependency failure path
            raise RuntimeError("VADER sentiment backend is unavailable.") from exc

        analyzer = SentimentIntensityAnalyzer()
        analyzer.lexicon.update(
            {
                "bullish": 2.4,
                "bearish": -2.4,
                "outperform": 1.9,
                "underperform": -1.9,
                "upgrade": 1.8,
                "downgrade": -1.8,
                "beats": 1.7,
                "beat": 1.5,
                "misses": -1.7,
                "miss": -1.5,
                "guidance raise": 1.8,
                "guidance cut": -1.8,
                "raises guidance": 1.9,
                "cuts guidance": -1.9,
                "buyback": 1.5,
                "layoffs": -1.6,
                "lawsuit": -1.4,
                "probe": -1.2,
                "investigation": -1.4,
                "record revenue": 1.8,
                "record profit": 1.9,
                "margin pressure": -1.6,
                "slump": -1.8,
                "surge": 1.7,
                "plunge": -2.0,
            }
        )
        self._vader_analyzer = analyzer
        return analyzer

    def _score_batch(self, texts: list[str]) -> list[float]:
        backend = str(self.config.backend or "auto").strip().lower()
        if backend not in {"auto", "transformers", "vader"}:
            raise RuntimeError(f"Unsupported local sentiment backend: {self.config.backend}")

        if backend in {"auto", "transformers"} and not self._transformer_disabled:
            try:
                classifier = self._build_transformer_pipeline()
                outputs = classifier(
                    texts,
                    batch_size=max(int(self.config.batch_size), 1),
                    truncation=True,
                    max_length=512,
                )
                return [_label_scores_to_sentiment(label_scores) for label_scores in outputs]
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._transformer_disabled = True
                self.logger.warning(
                    "Transformer sentiment backend unavailable, falling back to VADER: %s",
                    exc,
                )

        analyzer = self._build_vader_analyzer()
        return [float(analyzer.polarity_scores(text).get("compound", 0.0)) for text in texts]

    def _load_pending_articles(self) -> list[dict[str, Any]]:
        statement = select(
            NewsArticle.article_id,
            NewsArticle.headline,
            NewsArticle.summary,
        ).order_by(NewsArticle.created_at_source.asc(), NewsArticle.article_id.asc())
        if self.config.source:
            statement = statement.where(NewsArticle.source == self.config.source)
        if self.config.only_null_sentiment:
            statement = statement.where(NewsArticle.sentiment.is_(None))
        if self.config.limit is not None:
            statement = statement.limit(int(self.config.limit))

        with self.db_manager.engine.begin() as conn:
            rows = conn.execute(statement).fetchall()

        articles: list[dict[str, Any]] = []
        for row in rows:
            text = _combine_news_text(row.headline, row.summary)
            if not text:
                continue
            articles.append(
                {
                    "article_id": row.article_id,
                    "text": text,
                }
            )
        return articles

    def run(self) -> dict[str, Any]:
        articles = self._load_pending_articles()
        if not articles:
            return {
                "processed_rows": 0,
                "updated_rows": 0,
                "source": self.config.source,
                "model_name": self.config.model_name,
            }

        now = _now_utc()
        batch_size = max(int(self.config.batch_size), 1)
        total_updated = 0

        update_stmt = (
            update(NewsArticle)
            .where(NewsArticle.article_id == bindparam("target_article_id"))
            .values(
                sentiment=bindparam("target_sentiment"),
                updated_at=bindparam("target_updated_at"),
            )
        )

        for start_idx in range(0, len(articles), batch_size):
            batch = articles[start_idx : start_idx + batch_size]
            updates = [
                {
                    "target_article_id": item["article_id"],
                    "target_sentiment": sentiment_score,
                    "target_updated_at": now,
                }
                for item, sentiment_score in zip(
                    batch,
                    self._score_batch([item["text"] for item in batch]),
                    strict=True,
                )
            ]
            with self.db_manager.engine.begin() as conn:
                conn.execute(update_stmt, updates)
            total_updated += len(updates)
            self.logger.info(
                "Local news sentiment backfill progress: %s/%s",
                total_updated,
                len(articles),
            )

        return {
            "processed_rows": len(articles),
            "updated_rows": total_updated,
            "source": self.config.source,
            "model_name": self.config.model_name,
        }
