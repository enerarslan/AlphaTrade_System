from __future__ import annotations

from datetime import date

from quant_trading_system.data.free_gap_loader import (
    FINRA_MARKET_CODES,
    FreeGapBootstrapConfig,
    FreeGapDataBootstrapper,
)


def test_parse_ftd_rows_handles_extra_pipe_in_description(tmp_path) -> None:
    loader = FreeGapDataBootstrapper(
        FreeGapBootstrapConfig(output_root=tmp_path, sync_db=False),
    )
    text = (
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20210415|12345678|AAPL|100|APPLE INC | CLASS A|134.50\n"
    )

    rows = loader._parse_ftd_rows(text)

    assert rows == [
        {
            "SETTLEMENT DATE": "20210415",
            "CUSIP": "12345678",
            "SYMBOL": "AAPL",
            "QUANTITY (FAILS)": "100",
            "DESCRIPTION": "APPLE INC | CLASS A",
            "PRICE": "134.50",
        }
    ]


def test_discover_finra_short_sale_links_backfills_missing_dates(tmp_path, monkeypatch) -> None:
    loader = FreeGapDataBootstrapper(
        FreeGapBootstrapConfig(output_root=tmp_path, sync_db=False, short_sale_days=3),
    )
    html = (
        "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20260402.txt "
        "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260402.txt"
    )
    monkeypatch.setattr(loader, "_request_text", lambda url, provider: html)
    monkeypatch.setattr(
        loader,
        "_recent_trading_dates",
        lambda count: [date(2026, 3, 31), date(2026, 4, 1), date(2026, 4, 2)],
    )
    monkeypatch.setattr(
        loader,
        "_finra_short_sale_url_exists",
        lambda url: any(
            token in url
            for token in (
                "FNQCshvol20260401.txt",
                "CNMSshvol20260401.txt",
                "FNQCshvol20260331.txt",
            )
        ),
    )

    links = loader._discover_finra_short_sale_links()
    names = {url.rsplit("/", 1)[-1] for url, _trade_date, _market in links}

    assert "FNQCshvol20260402.txt" in names
    assert "CNMSshvol20260402.txt" in names
    assert "FNQCshvol20260401.txt" in names
    assert "CNMSshvol20260401.txt" in names
    assert "FNQCshvol20260331.txt" in names
    assert set(market for _url, _trade_date, market in links).issubset(set(FINRA_MARKET_CODES))
