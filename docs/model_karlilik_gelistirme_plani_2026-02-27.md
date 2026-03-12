# Model Karlilik Gelistirme Plani (Kurumsal Seviye)

Tarih: 2026-02-27
Kapsam: Sadece model gelistirme ve karlilik (guvenlik kapsam disi)

## 1) Kod Tabaninda Tespit Edilen Kritik Bosluklar

### P0 (once kapanacak)
1. Objective/etiket uyumsuzlugu riski
- `ModelManager` tarafinda bazi optimize akislari Sharpe odakli, fakat etiketler ikili sinif (0/1) olabiliyor. Bu, gercek net getiri yerine siniflama skoruna optimize olma riski tasir.
- Referanslar:
  - `quant_trading_system/models/model_manager.py:891`
  - `quant_trading_system/models/model_manager.py:924`
  - `quant_trading_system/models/target_engineering.py:204`

2. Horizon ve purge uyumsuzlugu
- Etiket ufku (or. h=5,20) ile CV tarafinda kullanilan `prediction_horizon` her yerde senkron degil.
- Referanslar:
  - `quant_trading_system/models/target_engineering.py:18`
  - `quant_trading_system/models/model_manager.py:1208`
  - `quant_trading_system/models/model_manager.py:1623`
  - `quant_trading_system/models/ensemble.py:363`

3. Alpha combiner tarafinda leakage riski
- Agirlik optimizasyonunda bazi yerlerde es-zamanli return kullanimi var; lag standardi tutarsiz.
- Referanslar:
  - `quant_trading_system/alpha/alpha_combiner.py:396`
  - `quant_trading_system/alpha/alpha_combiner.py:412`
  - `quant_trading_system/alpha/alpha_combiner.py:1153`

4. Backtest execution gerceklik acigi
- Latency/fill modeli var ama backtest engine'de zaman/fiyat etkisi tam tasinmiyor; partial fill davranisinda iyimserlik riski var.
- Referanslar:
  - `quant_trading_system/backtest/simulator.py:871`
  - `quant_trading_system/backtest/engine.py:830`
  - `quant_trading_system/backtest/engine.py:852`

### P1 (kisa vadede yuksek etki)
1. Cost modeli tutarliligi
- Target engineering spread/slippage/impact kullanirken, bazi eval akislari tek `assumed_cost_bps` ile hesapliyor.
- Referanslar:
  - `quant_trading_system/models/target_engineering.py:154`
  - `quant_trading_system/models/target_engineering.py:181`
  - `quant_trading_system/models/model_manager.py:1467`

2. Sample weight zinciri tam degil
- Uretilen sample weight her optimize/CV/final fit adiminda zorunlu degil.
- Referanslar:
  - `quant_trading_system/models/target_engineering.py:305`
  - `quant_trading_system/models/model_manager.py:843`
  - `quant_trading_system/models/model_manager.py:1286`

3. Cross-sectional feature dayaniklilik sorunu
- Eksik benchmark/universe durumunda sabit fallback degerleri (0/0.5/1) sinyal kalitesini bozabilir.
- Referanslar:
  - `quant_trading_system/features/cross_sectional.py:272`
  - `quant_trading_system/features/cross_sectional.py:373`
  - `quant_trading_system/features/cross_sectional.py:429`

4. Feature stability gate zorunlu degil
- Drift/temporal consistency kontrolu var ama pipeline kararina sert gate olarak bagli degil.
- Referanslar:
  - `quant_trading_system/features/feature_pipeline.py:617`
  - `quant_trading_system/features/feature_pipeline.py:1195`

### P2 (orta vade)
1. Regime-aware ensemble'da OOS agirlik ogrenimi guclendirilmeli.
- Referanslar:
  - `quant_trading_system/models/ensemble.py:605`
  - `quant_trading_system/models/ensemble.py:615`

2. Backtest-live objective birlesmeli.
- Referanslar:
  - `quant_trading_system/backtest/optimizer.py:545`
  - `quant_trading_system/trading/portfolio_manager.py:419`

## 2) Internet Arastirmasindan Cikan Prensipler (Kurumsal Egitim)

1. Overfitting kontrolu sadece tek Sharpe ile birakilmamali.
- Deflated Sharpe Ratio (DSR) ve PBO birlikte promotion gate'e baglanmali.
- Kaynaklar:
  - Bailey & Lopez de Prado, Deflated Sharpe Ratio (SSRN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
  - Bailey et al., Probability of Backtest Overfitting (SSRN): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

2. Data-snooping cezasi kurumsal standard olmalı.
- White Reality Check tipi yaklasimlar coklu deneme yanliligi riskini azaltir.
- Kaynak:
  - White (2000), Econometrica: https://doi.org/10.1111/1468-0262.00152

3. Zaman-serisi ayriminda leakage-safe CV zorunlu.
- Purged/walk-forward yaklasimi ve timestamp bazli split standard olmalidir.
- Kaynak:
  - scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

4. Olasilik kalibrasyonu canli performansi iyilestirir.
- Confidence threshold kararlarinda calibration (Platt/Isotonic) net etki verir.
- Kaynak:
  - scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html

5. Cross-sectional stock selection icin ranking objective degerli.
- Binary classifier yerine rank loss (NDCG/LambdaRank) karlilikla daha uyumlu olabilir.
- Kaynaklar:
  - XGBoost Learning to Rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
  - LightGBM params/objectives: https://lightgbm.readthedocs.io/en/latest/Parameters.html

6. ML asset pricing literaturunde capraz-kesit ogrenme ve regularizasyon on planda.
- Kaynak:
  - Gu, Kelly, Xiu (2020), RFS: https://doi.org/10.1093/rfs/hhaa009

7. Cok amacli HPO (return + drawdown + turnover) tek amacli skordan daha guvenli.
- Kaynak:
  - Optuna create_study: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html

## 3) 90 Gunluk Gelistirme Yol Haritasi

## Faz 1 (Gun 0-30): Objective ve Veri Sizin Aleyhinize Calismasin

Hedef: Model secimini gercek net getiriye hizalamak.

Teslimatlar:
1. Objective kilidi
- Classification modellerde ana objective: `logloss + calibration`; trading score sadece ikinci katman.
- Return/regression/ranking modellerde ana objective: cost-adjusted net return ve risk-adjusted utility.

2. Horizon/purge standardizasyonu
- `prediction_horizon == primary_label_horizon` zorunlulugu.
- CI gate: horizon uyumsuzsa train fail.

3. Sample weight end-to-end
- Target engineering'den gelen agirliklar hyperopt + CV + final fit akislarina zorunlu tasinsin.

4. Cost model tek kaynagi
- Spread/slippage/impact parametreleri train/backtest/holdout arasinda tek config objesiyle paylasilsin.

KPI/Gate:
- Holdout Sharpe > 0
- DSR >= 0.10
- PBO <= 0.45
- Holdout regime shift <= 0.35

## Faz 2 (Gun 31-60): Cross-Sectional Edge ve Gercekci Simulasyon

Hedef: Alpha kalitesini ve uygulanabilirligini birlikte yukari cekmek.

Teslimatlar:
1. Cross-sectional normalization katmani
- Her timestamp'te rank/zscore/robust scale.
- Benchmark eksiginde sabit fallback yerine `NaN + feature drop`.

2. Alpha combiner leakage temizligi
- Tum agirlik/IC hesaplarinda lag standardi teklesecek.

3. Backtest execution gerceklik upgrade
- Partial fill kalanlari ayni bar degil sonraki barlara tasinacak.
- Latency etkisi fiyat/zaman akisina baglanacak.
- Optimizer pencere backtest'lerinde tum execution parametreleri propagate edilecek.

KPI/Gate:
- CV -> Holdout Sharpe degradasyonu <= 35%
- Holdout symbol coverage >= 65%
- Holdout symbol p25 Sharpe >= -0.05
- Max drawdown <= 0.30

## Faz 3 (Gun 61-90): Kurumsal Model Ailesi ve Champion/Challenger

Hedef: Tek model bagimliligini azaltip kararlilik arttirmak.

Teslimatlar:
1. Model ailesi genisleme
- Binary classifier yanina ranker (LightGBM LambdaRank / XGBoost rank:ndcg) ve return regressor challengerlari ekle.
- Champion/Challenger matrisi: her horizon icin ayri leaderboard.

2. Rejim-adaptif agirliklarin OOS ogrenimi
- In-sample agirlik yerine walk-forward OOS regime-weight update.

3. Uc katmanli promotion paketi
- `Model Utility`, `Execution Robustness`, `Cross-Symbol Robustness` puanlari birlikte gecmeden production'a cikis yok.

KPI/Gate:
- 3 ay rolling holdout Sharpe istikrarli pozitif
- Underwater symbol ratio <= 45%
- Trade count gate saglaniyor (minimum aktivite)

## 4) Uygulama Sirasi (Net Backlog)

1. P0: Objective/label/horizon standartlarini kodda zorunlu hale getir.
2. P0: Sample weight ve cost modelini uc akista tekle.
3. P0: Leakage/fill/latency backtest aciklarini kapa.
4. P1: Cross-sectional fallback ve normalization refactor.
5. P1: Ranker challengerlarini train pipeline'a ekle.
6. P1: Champion/challenger promotion raporunu otomatiklestir.

## 5) Basariyi Nasil Olcecegiz

1. Arastirma metrikleri
- `mean_sharpe`, `holdout_sharpe`, `deflated_sharpe`, `pbo`, `holdout_worst_regime_sharpe`

2. Uygulanabilirlik metrikleri
- `turnover`, `max_drawdown`, `symbol_concentration_hhi`, `holdout_symbol_coverage_ratio`

3. Canliya yakinlik metrikleri
- backtest vs paper kayma farki (slippage/latency kaynakli)
- fill kalitesi ve reject orani

## 6) Notlar

- Bu plan model karliligina odaklidir; guvenlik kapsam disidir.
- Kod incelemesi agirlikli olarak `scripts/train.py`, `quant_trading_system/models/*`, `quant_trading_system/features/*`, `quant_trading_system/alpha/*`, `quant_trading_system/backtest/*`, `quant_trading_system/trading/*` dosyalarina dayanir.

## 7) Uygulama Durumu (2026-02-27)

Tamamlananlar:
1. P0 objective/metric/horizon standartlari (tamamlandi)
- Trading metric ile classification/regression uyumu kodda zorunlu hale getirildi.
- `prediction_horizon` tum CV/train optimize akislari icin normalize edilip `gap/purge` ile senkronlandi.
- Dosyalar:
  - `quant_trading_system/models/model_manager.py`
  - `tests/unit/test_models.py`

2. P0 sample weight end-to-end (tamamlandi)
- `train_model`, `cross_validate`, `nested_cross_validate`, `optimize_and_train`, `HyperparameterOptimizer` boyunca `sample_weights` zinciri eklendi.
- Dosyalar:
  - `quant_trading_system/models/model_manager.py`
  - `tests/unit/test_models.py`

3. P0 backtest execution gerceklik upgrade (tamamlandi)
- Pending order isleme snapshot'a alindi; partial remainder ayni bar loop'unda tekrar fill edilmiyor.
- Market simulator latency sonucu olusan fill'ler execution timestamp gelene kadar defer ediliyor.
- Window optimize backtest config'lerinde execution parametreleri (commission/slippage/mode vb.) preserve ediliyor.
- Dosyalar:
  - `quant_trading_system/backtest/engine.py`
  - `quant_trading_system/backtest/optimizer.py`
  - `tests/unit/test_backtest_engine_risk.py`
  - `tests/unit/test_backtest.py`

4. P1 cross-sectional fallback + normalization refactor (tamamlandi)
- Benchmark/universe eksiginde 0/0.5/1 sabit fallback yerine `NaN` tabanli fail-soft davranis.
- Zaman hizalama ve output sanitization guclendirildi; robust normalization katmani eklendi.
- Dosyalar:
  - `quant_trading_system/features/cross_sectional.py`
  - `tests/unit/test_features.py`

5. P0 alpha combiner lag/leakage standardizasyonu (tamamlandi)
- Tum return-tabanli alpha agirliklandirma akislari tek lag helper fonksiyonunda birlestirildi.
- `OptimizedWeighter` ve `get_alpha_stats(IC)` ayni lag standardina gecirildi.
- `CombinerConfig.return_lag` eklendi; tum weighter olusturma akislarina propagate edildi.
- Dosyalar:
  - `quant_trading_system/alpha/alpha_combiner.py`
  - `tests/unit/test_alpha.py`

6. P0 cost model single-source (tamamlandi)
- Canonical `TradingCostModel` olusturuldu ve model manager + target engineering + train script maliyet hesaplari bu modelle tekillestirildi.
- Legacy `assumed_cost_bps` arayuzleri backward-compatible tutuldu, ancak icerde canonical modele resolve ediliyor.
- Dosyalar:
  - `quant_trading_system/models/trading_costs.py`
  - `quant_trading_system/models/model_manager.py`
  - `quant_trading_system/models/target_engineering.py`
  - `scripts/train.py`
  - `tests/unit/test_models.py`

7. P1 alpha combiner OOS online agirlik update (tamamlandi)
- `expanding` ve `rolling` modlari ile OOS agirlik update akisi eklendi.
- OOS update blend, pencere ve minimum gozlem esikleri config uzerinden yonetiliyor.
- Aagirlik history ve OOS observation sayaçlari eklendi.
- Dosyalar:
  - `quant_trading_system/alpha/alpha_combiner.py`
  - `tests/unit/test_alpha.py`

8. P1 backtest objective + replay gate cost model entegrasyonu (tamamlandi)
- `StrategyOptimizer` objective skorlari canonical maliyet modeline bagli regularization ile cost-aware hale getirildi.
- Replay SLO degerlendirmesine expected-vs-actual execution cost ratio gate'i eklendi.
- Replay outcome icine expected execution cost metadatasi yaziliyor.
- Dosyalar:
  - `quant_trading_system/backtest/optimizer.py`
  - `quant_trading_system/backtest/replay.py`
  - `tests/unit/test_backtest.py`
  - `tests/unit/test_backtest_replay.py`

9. Faz 3 ranker challenger + horizon-bazli champion/challenger (tamamlandi)
- `lightgbm_ranker` model tipi institutional train pipeline'a eklendi (GPU preflight + parser + all-model dispatch dahil).
- Ranker fit akislarinda timestamp-query group olusturma ve oos/cv/final fit akislarina group propagation eklendi.
- Champion/challenger secimi horizon-bazli leaderboard uretecek sekilde genislatildi; snapshot icine horizon bazli champion/challenger bloklari yaziliyor.
- Dosyalar:
  - `scripts/train.py`
  - `tests/unit/test_train_script.py`

10. Faz 3 uc-katmanli promotion hard-gate (tamamlandi)
- Promotion karari artik `model_utility`, `execution_robustness`, `cross_symbol_robustness` katmanlarinin tumu gecilmeden `ready_for_production` olmuyor.
- Layer sonuclari validation report + model card + deployment plan + benchmark matrix akislarina tasindi.
- Champion seciminde layered gate uygunlugu zorunlu hale getirildi.
- Dosyalar:
  - `scripts/train.py`
  - `tests/unit/test_train_script.py`

11. Faz 3 return-regressor challenger hatti (tamamlandi)
- `xgboost_regressor` ve `lightgbm_regressor` model tipleri train pipeline'a eklendi.
- Regressor hedef secimi `triple_barrier_net_return` (fallback: `forward_return_h{primary}`) olacak sekilde target-mode yonetimi eklendi.
- CV/Optuna/final-fit akislarinda binary-sinif zorunlulugu regression modellerde kaldirildi; directional metrikler regresyon icin sign-bazli normalize edildi.
- Dosyalar:
  - `scripts/train.py`
  - `tests/unit/test_train_script.py`

12. White Reality Check / data-snooping gate entegrasyonu (tamamlandi)
- Block-bootstrap tabanli White Reality Check benzeri istatistiksel dogrulama eklendi (`white_reality_stat`, `white_reality_pvalue`, yorum).
- Promotion validation gate'lerine `min_white_reality_stat` ve `max_white_reality_pvalue` eklendi; model utility katmanina baglandi.
- Promotion package / model card / governance score akislarina yeni metrikler propagate edildi.
- Dosyalar:
  - `quant_trading_system/models/statistical_validation.py`
  - `scripts/train.py`
  - `tests/unit/test_statistical_validation.py`
  - `tests/unit/test_train_script.py`

Test durumu:
- `pytest tests/unit/test_backtest_engine_risk.py -q` -> PASS
- `pytest tests/unit/test_backtest.py -k "StrategyOptimizerSelection or BacktestConfig" -q` -> PASS
- `pytest tests/unit/test_features.py -k "CrossSectionalFeatures" -q` -> PASS
- `pytest tests/unit/test_models.py -k "HyperparameterOptimizer or ModelManagerInstitutionalValidation" -q` -> PASS
- `pytest tests/unit/test_alpha.py -k "AlphaCombiner" -q` -> PASS
- `pytest tests/unit/test_target_engineering.py -q` -> PASS
- `pytest tests/unit/test_backtest_replay.py -q` -> PASS
- `pytest tests/unit/test_train_script.py -q` -> PASS
- `pytest tests/unit/test_statistical_validation.py -q` -> PASS
