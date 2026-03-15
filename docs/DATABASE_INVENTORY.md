# AlphaTrade Database Inventory

> Son guncelleme: 2026-03-15
> Veritabani: TimescaleDB (PostgreSQL 15) @ 127.0.0.1:5433 / quant_trading
> Toplam boyut: ~1.36 GB | 25 tablo | 10.5M+ satir

---

## 1. OHLCV Fiyat Verileri (`ohlcv_bars`)

**Boyut:** 1.04 GB | **Toplam:** 10,071,430 satir

| Timeframe | Satir | Sembol | Baslangic | Bitis |
|-----------|-------|--------|-----------|-------|
| 1Min | 7,592,231 | 48 | 2024-03-14 | 2026-03-13 |
| 15Min | 1,583,910 | 48 | 2021-03-15 | 2026-03-13 |
| 1Day | 895,289 | 250 | 2011-03-18 | 2026-03-13 |

**1Min/15Min sembolleri (48 adet):**
AAPL, ABBV, AMD, AMZN, BA, BAC, BLK, C, CAT, COP, CRM, CVX, DIS, GE, GLD, GOOGL, GS, HD, HON, INTC, IWM, JNJ, JPM, KO, LLY, MA, MCD, META, MRK, MS, MSFT, NFLX, NKE, NVDA, PEP, PFE, QQQ, SBUX, SPY, T, TLT, TSLA, UNH, UPS, V, VZ, WMT, XOM

**1Day:** 250 sembol (S&P 250 evren), hepsi 2011-03-18'den 2026-03-13'e kadar tam (3,769 bar/sembol).

**Veri kalitesi:**
- NULL close veya volume yok
- high < low anomalisi yok
- 1,379 zero-volume bar (hepsi 1Day - tatil/yarim gun kayitlari)

---

## 2. Alternatif Veri

### SEC Filings (`sec_filings`)
- **Boyut:** 247 MB | **Satir:** 308,752
- **Aralik:** 1996-01-19 → 2026-03-13
- **Sembol:** 244
- Icerik: accession_number, form tipi, filing/report tarih, URL'ler, metadata (JSONB)

### Haber Verileri (`news_articles`)
- **Boyut:** 52 MB | **Satir:** 38,041
- **Aralik:** 2025-03-14 → 2026-03-14 (son 1 yil)
- Icerik: headline, summary, URL, sentiment skoru, symbols (JSONB), kaynak

### Corporate Actions (`corporate_actions`)
- **Boyut:** 6.5 MB | **Satir:** 25,884
- **Aralik:** 1962-01-16 → 2026-03-13
- **Sembol:** 233
- Icerik: dividendlar, split'ler, ex-date bazli

### Short Sale Volumes (`short_sale_volumes`)
- **Boyut:** 1.7 MB | **Satir:** 9,914
- **Aralik:** 2026-03-02 → 2026-03-13 (sadece son 2 hafta)
- **Sembol:** 248
- Icerik: short_volume, short_exempt_volume, total_volume, market

### Fails to Deliver (`fails_to_deliver`)
- **Boyut:** 720 KB | **Satir:** 3,536
- **Aralik:** 2025-12-15 → 2026-02-13 (son ~2 ay)
- **Sembol:** 248
- Icerik: settlement_date, CUSIP, quantity, price

### Earnings Events (`earnings_events`)
- **Boyut:** 664 KB | **Satir:** 980
- **Sembol:** 245
- Not: reported_date alani bos (tarihler NULL)

### Fundamental Snapshots (`fundamental_snapshots`)
- **Boyut:** 792 KB | **Satir:** 250
- **Tarih:** 2026-03-14 (tek gunluk snapshot)
- **Sembol:** 250
- Icerik: as_of_date bazli temel veriler

---

## 3. Makroekonomik Veriler

### Macro Observations (`macro_observations`)
- **Boyut:** 18 MB | **Satir:** 60,890 | **Seri:** 14

| Seri ID | Aciklama | Gozlem | Baslangic | Bitis |
|---------|----------|--------|-----------|-------|
| DGS10 | 10-Year Treasury Yield | 16,748 | 1962-01-02 | 2026-03-12 |
| DGS2 | 2-Year Treasury Yield | 12,988 | 1976-06-01 | 2026-03-12 |
| VIX | CBOE Volatility Index | 9,142 | 1990-01-02 | 2026-03-13 |
| VVIX | VIX of VIX | 4,977 | 2006-03-06 | 2026-03-13 |
| GVZ | Gold Volatility Index | 4,144 | 2009-09-18 | 2026-03-13 |
| OVX | Oil Volatility Index | 4,144 | 2009-09-18 | 2026-03-13 |
| VIX9D | 9-Day VIX | 3,820 | 2011-01-04 | 2026-03-13 |
| PAYEMS | Nonfarm Payrolls | 1,046 | 1939-01-01 | 2026-02-01 |
| CPIAUCSL | Consumer Price Index | 950 | 1947-01-01 | 2026-02-01 |
| UNRATE | Unemployment Rate | 938 | 1948-01-01 | 2026-02-01 |
| FEDFUNDS | Federal Funds Rate | 860 | 1954-07-01 | 2026-02-01 |
| RSAFS | Retail Sales | 409 | 1992-01-01 | 2026-01-01 |
| DGORDER | Durable Goods Orders | 408 | 1992-02-01 | 2026-01-01 |
| GDPC1 | Real GDP | 316 | 1947-01-01 | 2025-10-01 |

### Macro Vintage Observations (`macro_vintage_observations`)
- **Boyut:** 1.8 MB | **Satir:** 8,625 | **Seri:** 9
- **Aralik:** 2011-01-01 → 2026-03-12
- Point-in-time revizyon takibi (look-ahead bias onleme icin)

---

## 4. Referans Verileri

### Security Master (`security_master`)
- **Satir:** 250 | **Boyut:** 480 KB
- Icerik: symbol, name, exchange, asset_type, sector, industry, market_cap, shares_outstanding, CIK, IPO tarihi
- Trading evreninin referans tablosu

---

## 5. Feature Store (`features`)

- **Boyut:** 2 MB | **Satir:** 20,133
- **Sembol:** Sadece AAPL
- **Aralik:** 2025-01-02 → 2026-02-19
- **Durum:** EKSIK - 249 sembol icin feature hesaplanmamis
- Aksiyon: `python main.py features compute` ile tum semboller icin hesaplanmali

---

## 6. Trading & Operations Tablolari (BOS)

Asagidaki tablolar henuz bos - canli/paper trading veya backtest calistirilmamis:

| Tablo | Amac | Satir |
|-------|------|-------|
| orders | Emir kayitlari | 0 |
| trades | Gerceklesen islemler | 0 |
| positions | Acik pozisyonlar | 0 |
| position_history | Pozisyon gecmisi | 0 |
| signals | Uretilen sinyaller | 0 |
| model_predictions | Model tahminleri | 0 |
| daily_performance | Gunluk performans | 0 |
| alerts | Sistem alarmlari | 0 |
| risk_events | Risk olaylari | 0 |
| system_logs | Sistem loglari | 0 |
| stock_quotes | Canli kotasyonlar | 0 |
| stock_trades | Canli islem akisi | 0 |
| trade_log | Islem logu | 0 |

---

## 7. Altyapi Notlari

- **Hypertable:** Olusturulmamis - tum tablolar plain PostgreSQL heap. ohlcv_bars icin TimescaleDB hypertable donusumu yapilabilir.
- **Baglanti:** `postgresql://postgres:***@127.0.0.1:5433/quant_trading`
- **Tablo sayisi:** 25
- **Toplam disk:** ~1.36 GB
