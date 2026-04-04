from quant_trading_system.data.news_sentiment import (
    _combine_news_text,
    _label_scores_to_sentiment,
)


def test_combine_news_text_prefers_non_empty_parts() -> None:
    assert _combine_news_text("Headline", "Summary body") == "Headline\n\nSummary body"
    assert _combine_news_text("Headline", None) == "Headline"
    assert _combine_news_text(None, "Summary body") == "Summary body"


def test_label_scores_to_sentiment_returns_signed_probability_gap() -> None:
    scores = [
        {"label": "positive", "score": 0.72},
        {"label": "neutral", "score": 0.18},
        {"label": "negative", "score": 0.10},
    ]

    assert _label_scores_to_sentiment(scores) == 0.62
