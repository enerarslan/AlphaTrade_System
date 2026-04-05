# Production Readiness Audit - 2026-04-05

## Scope

Bu rapor, mevcut Git workspace'i ile WSL tarafindaki gercek run artefaktlarini birlikte inceleyerek hazirlandi.

- Kod workspace'i: `C:\Users\ener\Desktop\AlphaTrade`
- WSL run dizini: `/root/AlphaTrade_wsl`
- WSL artefakt kaynaklari:
  - `/root/AlphaTrade_wsl/logs/*.stdout.log`
  - `/root/AlphaTrade_wsl/models/*_artifacts.json`
  - `/root/AlphaTrade_wsl/models/benchmarks/training_matrix_*.json`

Kritik not: `/root/AlphaTrade_wsl` bir Git checkout degil. `.git/` yok. Bu tek basina production promotion icin ciddi provenance acigi demek.

## Executive Summary

Sonuc net: WSL'deki hicbir `runD*` veya `runE0` adayi production ready degil.

Ana neden tek bir metrik degil; sistemik:

1. Tum ciddi adaylar `max_pbo` gate'inde fail oluyor.
2. "Production adayi" gibi bakilan run'lar gercekte `research` profili ile calismis.
3. Nested CV stabilite kapisi fail olsa bile egitim kodu unstable adaya fallback yapiyor.
4. Ranker tarafinda skorlar `query_percentile` ile pseudo-probability'ye cevriliyor ve probability calibration kapali.
5. Objective icinde negatif skew cezasi bazi run'larda sonucu domine ediyor.
6. Feature engineering runtime'i pahali; ozellikle `statistical` ve `cross_sectional` gruplari duvar suresi yiyor.
7. Operasyonel metrikler operatoru yaniltabilecek sekilde birden fazla activity tanimini ayni isim etrafinda karistiriyor.
8. Ortam hardening'i eksik: dev audit secret fallback var, file fallback hala mevcut, `--require-gpu` feature tarafinda gercek GPU readiness garanti etmiyor.

## Best Current Candidate Read

- `D1`: en guclu ham arastirma sonucu. Ama cok agresif, symbol tail zayif ve PBO fail.
- `D3`: en dengeli ranker arastirma adayi. Symbol p25 ve regime dayanikliligi iyi. Ama holdout sharpe negatif, consistency fail ve PBO cok kotu.
- `D1f`: `short_sale + ftd + symbol priors` yonu dogru. Aktiviteyi ciddi dusuruyor ve symbol tail'i D1'e gore iyilestiriyor. Ama skew cezasi yuzunden risk-adjusted skor cok kotu.
- `E0`: calibrated XGBoost baseline'i olarak degerli. Ama robustluk halen production seviyesinde degil.

Pratik sonuc: bir sonraki run icin en iyi temel yon `D1f/D2f` ailesi. Ama once training contract'i ve promotion gate'leri duzeltilmeli. Sadece yeni hiperparametre denemek yeterli olmayacak.

## Run Comparison

Asagidaki tablo WSL artefaktlarindan okunmustur. `Gate Active Rate`, validation gate tarafinda kullanilan etkili holdout aktif sinyal oranidir; ham threshold hit rate degildir.

| Run | Mean Sharpe | Risk Adj | PBO | Holdout Sharpe | Symbol P25 | Worst Regime | Gate Active Rate | Failed Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D0 | 0.341 | -2.922 | 0.878 | -0.340 | -0.774 | 1.573 | 0.330 | `min_sharpe_ratio`, `risk_adjusted_positive`, `max_pbo`, `max_white_reality_pvalue`, `min_holdout_sharpe`, `min_holdout_symbol_p25_sharpe` |
| D1 | 1.524 | 0.868 | 0.522 | 1.753 | -0.620 | 1.881 | 0.344 | `max_pbo`, `min_holdout_symbol_p25_sharpe` |
| D1b | 0.558 | -1.694 | 0.590 | 1.108 | 0.072 | 1.358 | 0.283 | `risk_adjusted_positive`, `max_pbo`, `max_white_reality_pvalue` |
| D1c | 0.438 | -2.065 | 0.520 | 1.335 | 1.355 | -0.346 | 0.242 | `min_sharpe_ratio`, `risk_adjusted_positive`, `max_pbo`, `min_holdout_regime_sharpe` |
| D1f | 0.139 | -3.834 | 0.502 | 2.315 | 0.187 | 1.115 | 0.158 | `min_sharpe_ratio`, `risk_adjusted_positive`, `max_pbo` |
| D1g | 0.459 | -2.574 | 0.710 | 2.342 | -1.310 | 1.015 | 0.510 | `min_sharpe_ratio`, `risk_adjusted_positive`, `max_pbo`, `max_white_reality_pvalue`, `min_holdout_symbol_p25_sharpe` |
| D2 | 1.077 | 0.851 | 0.580 | 0.821 | -0.923 | -0.776 | 0.291 | `max_pbo`, `min_holdout_regime_sharpe`, `min_holdout_symbol_p25_sharpe` |
| D3 | 0.612 | -0.785 | 0.711 | -0.656 | 0.844 | 1.633 | 0.277 | `risk_adjusted_positive`, `max_pbo`, `min_holdout_sharpe`, `holdout_sharpe_consistency` |
| E0 | 0.892 | 0.649 | 0.549 | 1.396 | -0.145 | -0.497 | 0.278 | `max_pbo`, `min_holdout_regime_sharpe`, `min_holdout_symbol_p25_sharpe` |

## Primary Findings

### 1. Promotion adayi zannedilen run'lar research profile ile calismis

`scripts/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh` icinde `--training-profile research` hardcoded. Bu da tum bu WSL run'larini dogal olarak hizli iterasyon moduna sokuyor.

Kod referanslari:

- `scripts/launch_wave1_wsl_clean_ranker_scope_safe_h12.sh:56`
- `scripts/train.py:154-166`

Etkisi:

- Daha dusuk `n_trials`
- Daha az nested split
- SHAP ve meta-labeling gevsetmeleri
- Bazi promotion evidence zorunluluklarinin gevsemesi

Sonuc: bu run'lardan hicbiri "production-ready challenger" olarak yorumlanmamali.

### 2. Stable outer-fold yoksa kod yine de unstable adayi seciyor

WSL log kaniti, `D1f` icin:

- `Outer fold 1: rejected due to unstable Optuna surface (ratio=2.5388 > cap=1.2500).`
- `Outer fold 2: rejected due to unstable Optuna surface (ratio=2.3129 > cap=1.2500).`
- `No stable outer-fold candidates passed stability gate; falling back to best unstable candidate.`

Kod referanslari:

- `scripts/train.py:6784`
- `scripts/train.py:6862-6863`

Bu davranis research icin kabul edilebilir olabilir. Production aday secimi icin kabul edilemez. Promotion path'inde hard fail olmali.

### 3. En kalici blocker PBO

Butun ciddi adaylar `max_pbo` kapisinda fail:

- D1: `0.522`
- D1f: `0.502`
- D2: `0.580`
- D3: `0.711`
- E0: `0.549`

Threshold `0.45`. Bu degisken, su anda model seciminin overfit ve secim yanliligina acik kaldigini gosteren en kuvvetli ortak sinyal.

### 4. Ranker probability contract'i ekonomik karar almaya uygun degil

Ranker tarafinda iki temel problem var:

1. Skorlar `query_percentile` ile normalize ediliyor.
2. Ranker modeller icin probability calibration tamamen kapali.

Kod referanslari:

- `scripts/train.py:3911-3926`
- `scripts/train.py:8925-8933`

`D1f` artefakt kaniti:

- `ranker_score_normalization = query_percentile`
- OOF raw dagilim min/max: `0.04545 / 0.95455`
- Holdout raw dagilim min/max: `0.04545 / 0.95455`
- Raw ve calibrated dagilimlar birebir ayni

Bu, 11 isimlik panelde skorlarin kaba ve ayrik bir merdivene donustugunu gosteriyor. Esiklerin davranisi gercek olasilik degil, sadece cross-sectional ranking.

Sonuc:

- Threshold tuning kirilgan oluyor
- Activity davranisi zor yorumlaniyor
- Calibration olmayinca expected-edge policy'nin ekonomik anlami zayifliyor

### 5. Activity metrikleri operatoru yaniltiyor

`D1f` ayni anda su sayilari uretiyor:

- `holdout_raw_threshold_hits_active_rate = 0.229`
- `holdout_execution_signal_activity_active_signal_rate = 0.142`
- `effective_holdout_active_signal_rate_metric = 0.158`
- `holdout_active_signal_rate = 0.526`
- `expected_edge_holdout_selected_rate = 0.025`

Bu sayilar teknik olarak farkli katmanlari olcuyor, ama isimlendirme operator icin net degil. Su an "aktiflik" tek bir kavram gibi gozukup aslinda bes farkli davranisi karistiriyor:

- raw threshold hit
- execution-band sonrası sinyal
- gate icin kullanilan efektif aktiflik
- portfolio/trade mask aktifligi
- expected-edge policy coverage

Production raporlamasi icin bunlar ayrik ve standardize edilmeden operator karari saglikli olmaz.

### 6. Objective icinde skew cezasi bazi run'lari domine ediyor

Kod referansi:

- `scripts/train.py:11526`

Ceza formu:

- `skew_penalty = -objective_weight_skew * max(0.0, -skew)`

Yani negatif skew teorik olarak sinirsiz ceza alabiliyor.

`D1f` kaniti:

- `mean_return_skew = -25.133`
- `mean_objective_skew_penalty = -3.505`
- `mean_risk_adjusted_score = -3.834`

Bu run'da risk-adjusted skorun ana yikici bileseni skew cezasi. `D1g` icin de benzer problem var. Bu kadar buyuk skew mutlak degerleri ya asiri duyarli estimator'a ya da return stream icinde tail outlier'larin objective'i asiri bozduguna isaret ediyor.

### 7. Feature selection faydali ama observability bozuk

WSL loglari feature selection'in gercekte binding oldugunu gosteriyor:

- D1: `320 -> 103`
- D1b: `321 -> 112`
- D1c: `320 -> 80`
- D1f: `331 -> 109`
- D1g: `347 -> 109`
- D2: `320 -> 96`

Ama final artefaktlarda `feature_selection_binding = 0.0` gorunebiliyor; cunku son refit asamasinda tekrar daralmiyor.

Bu operator acisindan yaniltici. Iki farkli metrik lazim:

- `development_feature_selection_binding`
- `final_refit_feature_selection_binding`

Su an tek sayi, gercekte uygulanan secimin siddetini gizliyor.

### 8. Asil runtime bottleneck feature engineering

`D1f` log kaniti:

- `Phase 2: Computing features...`
- `Snapshot-only pipeline completed in 1890.7s`
- `Training completed in 3798.5s`

Per-symbol surelere bakinca:

- `technical`: yaklasik `14-16s`
- `statistical`: yaklasik `62-68s`
- `microstructure`: yaklasik `9-10s`
- `cross_sectional`: yaklasik `61-67s`

Kod tarafinda nedenler net:

- `quant_trading_system/features/statistical.py:293-408`
- `quant_trading_system/features/statistical.py:1431-1440`
- `quant_trading_system/features/cross_sectional.py:364-480`
- `quant_trading_system/features/cross_sectional.py:594-799`
- `quant_trading_system/features/cross_sectional.py:875-928`
- `quant_trading_system/features/cross_sectional.py:1076-1092`

Mevcut implementasyon:

- rolling window'lari Python loop'lariyla hesapliyor
- cross-sectional hesaplari her symbol icin tekrar tekrar yapiyor
- universe alignment ve rolling correlation/PCA islerini yeterince ortaklastirmiyor

Bu alan, bir sonraki buyuk run'dan once optimize edilmesi gereken birinci performans hedefi.

### 9. GPU readiness semantigi yaniltici

Launcher `--require-gpu` gonderiyor, ama log tekrar tekrar soyle diyor:

- `GPU requested but cuDF not available, using parallel CPU`

Kod referansi:

- `scripts/train.py:1001-1077`

Sorun:

- GPU check model backend seviyesinde
- feature engineering tarafinda cuDF/cupy readiness garanti edilmiyor

Sonuc: operator "GPU ile calistim" sanabilir ama feature duvar suresi CPU'da kalir.

### 10. Production hardening aciklari halen acik

Kod referanslari:

- `quant_trading_system/monitoring/dashboard.py:2533-2544`
- `quant_trading_system/data/data_access.py:10`
- `quant_trading_system/data/data_access.py:164-182`
- `quant_trading_system/data/data_access.py:297-319`

WSL log kaniti:

- `DASHBOARD_AUDIT_SECRET is not set, using development-only fallback secret for signed audit records.`

Production-ready bir run icin bunlar blocker:

- dev audit secret fallback
- DB disi file fallback semantigi
- provenance'siz WSL mirror

### 11. Numerical warning'ler henuz yeterince governance'a yansimiyor

WSL loglari tekrarli su warning'leri veriyor:

- `invalid value encountered in divide`
- `np.corrcoef` icinde invalid division
- `technical.py` ve `microstructure.py` kaynakli divide-by-range warning'leri

Bu warning'ler su an daha cok log gürültüsü. Bunlarin feature-level null-rate, clipped-rate, finite-rate metriklerine donusturulmesi gerekir. Aksi halde "gecti" denilen run aslinda zayif sayisal stabilite ile geciyor olabilir.

## What Should Change Before The Next Run

### A. Promotion contract'i sertlestir

Zorunlu:

1. Production adayi icin `--training-profile promotion` kullan.
2. Promotion path'inde unstable outer-fold fallback'i hard fail yap.
3. Promotion run, yalnizca `pre_promotion_ready = true` olan arastirma adayindan baslayabilsin.
4. Artefakta `git_commit`, `git_dirty`, `environment_lock_hash`, `training_host`, `wsl_distro` yaz.
5. `/root/AlphaTrade_wsl` yerine gercek Git checkout veya `git worktree` kullan.

### B. Metrik contract'ini duzelt

Zorunlu:

1. `active_rate` adini tek bir kavram olarak kullanma.
2. Ayrik metrikler yaz:
   - `raw_threshold_active_rate`
   - `execution_signal_active_rate`
   - `effective_gate_active_rate`
   - `portfolio_trade_active_rate`
   - `expected_edge_selected_rate`
3. Feature selection icin development ve final refit binding metriklerini ayir.
4. Probability calibration state'ini her model ailesi icin acik yaz.

### C. Modeling yonu

Bir sonraki arastirma adayi icin onerilen temel:

- `D2f/D1f` ailesi
- `sec_filings + corporate_actions + short_sale + ftd`
- `symbol priors` acik
- haber (`news`) su asamada kapali kalsin

Neden:

- `D1g` haber eklenince symbol tail bozuluyor
- `D1f` aktiviteyi ve symbol tail'i D1'e gore daha saglikli yone itiyor

Zorunlu model degisiklikleri:

1. Ranker icin calibration stratejisi ekle veya calibrated classifier baseline'i promotion challenger olarak tut.
2. `query_percentile` normalization'in economic edge ile iliskisini netlestir; gerekirse ranker'i probability-like degil score-based execution contract ile kullan.
3. Skew cezasina cap koy veya robust skew estimator kullan.
4. Tail penalty ile skew penalty'nin birlikte objective'i domine etmesini engelle.

### D. Performans optimizasyonu

Bir sonraki buyuk run'dan once yapilmasi gerekenler:

1. Cross-sectional feature'lari symbol-bazli tekrar hesaplamak yerine universe-bazli bir kez hesapla, sonra symbol'e dagit.
2. `statistical.py` icindeki rolling loop'lari vectorized/Numba/Polars tabanli hale getir.
3. Universe alignment cache'lerini per-window yeniden kullan.
4. `--require-gpu` kullaniliyorsa feature tarafinda da gercek GPU readiness fail-fast olsun.
5. Numerical warning'leri sadece loglama; feature quality metrics olarak say.

## Recommended Next Run Gate

Bir sonraki "production adayi" run ancak su sartlarla anlamli olur:

- Gercek Git checkout/worktree kullaniliyor
- `training_profile = promotion`
- Stable outer fold bulundu
- `max_pbo <= 0.45`
- `risk_adjusted_positive` pass
- `min_holdout_symbol_p25_sharpe` pass
- `min_holdout_regime_sharpe` pass
- `DASHBOARD_AUDIT_SECRET` set
- DB-only runtime semantigi acik
- Feature GPU readiness veya CPU fallback durumu artefakta acik
- Replay ve health-check ile tekrar dogrulama yapildi

Bu checklist saglanmadan alinacak yeni "iyi skor" sadece bir sonraki research iteration olur; production candidate olmaz.

## Validation Performed For This Audit

Bu audit sirasinda:

- WSL log ve artefaktlari okundu
- kritik training/evaluation kod patikalari incelendi
- asagidaki testler mevcut workspace'te calistirildi ve gecti:
  - `pytest tests/unit/test_train_script.py -k "cross_sectional_row_budget or feature_selection_records_detailed_audit or promotion_profile_rejects_optional_governance_disables" -v`
  - `pytest tests/unit/test_features.py -k "adaptive_sampling_step_scales_large_datasets or cross_sectional_calculator_preserves_bounded_percentile_features" -v`

Calistirilmayanlar:

- tam replay
- integration websocket/data pipeline testleri
- `python main.py health check --full`

Dolayisiyla bu belge production hardening ve modeling diagnosis dokumanidir; final infra sign-off belgesi degildir.

## Bottom Line

Bir sonraki run icin en dogru strateji:

1. Once training contract ve observability contract'ini production seviyesine getir.
2. Sonra `D1f/D2f` yonunu, calibration ve skew-penalty duzeltmeleriyle tekrar dene.
3. Feature engineering performansini optimize etmeden buyuk grid search'e geri donme.
4. WSL mirror yerine provenance'i olan gercek bir checkout uzerinden calis.

Bugunku durumda "daha cok trial" degil, "daha dogru contract + daha net metrics + daha az overfit" gerekiyor.
