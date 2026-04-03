# Wave1 Model Training Plan

**Tarih:** 2026-04-02  
**Kapsam:** `wave1` cok-sembollu intraday model gelistirme, artifact-driven backtest, replay ve paper trading gecis plani  
**Baglam:** `r1`, `r2` ve `r3` ranker research run'lari core stack'i ileri tasisa da production gate'lerini gecemedi. Bu dokuman, guncel run karsilastirmasi ve mevcut repo davranisina gore onerilen training sirasini tanimlar.

## 1. Karar Ozeti

Ana yol su olmali:

1. **Primary model:** `lightgbm_ranker`
2. **Primary challenger:** `xgboost`
3. **Sanity baseline:** `elastic_net`
4. **Opsiyonel family check:** `lightgbm` classifier
5. **Simdilik yok:** `lstm`, `transformer`, `tcn`, production ensemble

Bunun nedeni:

- Problemimiz **cross-sectional intraday ordering** problemi. Bu yuzden `lightgbm_ranker` yapisal olarak en dogru ilk aday.
- Yine de tek modele guvenmek dogru degil. Ayni snapshot uzerinde en az bir guclu challenger ve bir basit baseline gerekir.
- Derin modeller icin veri hacmi, operasyon maliyeti ve debug karmasikligi su asamada gereginden yuksek.
- Ensemble ancak **iki bagimsiz model familyasi promotion seviyesini ayri ayri gecerse** anlamli olur.

Kisa cevap:

- **LightGBM tek basina yeterli degil.**
- **LightGBM ranker primary olmali, ama yaninda mutlaka `xgboost` ve `elastic_net` train edilmelidir.**

## 1.1 2026-04-03 Guncellemesi

Su anki en son iki anlamli run:

- `r2_covtailfix_h12`
- `r3_refcurated_iso_h12`

`r3`, curated reference subset ile su alanlarda iyilesme sagladi:

- holdout sharpe yukseltti
- worst-symbol sharpe'i belirgin toparladi
- symbol concentration'i dusurdu
- median symbol sharpe'i yukseltti

Ama production tarafinda daha kritik alanlarda geri gitti:

- `mean_sharpe` dustu
- `mean_risk_adjusted_score` belirgin kotulesti
- `pbo` kotulesti
- `white_reality_check_pvalue` fail oldu
- `holdout_symbol_sharpe_p25` hala fail

Ana yorum:

- curated references tam bir cozum degil
- tail risk yalnizca semboller arasinda yer degistirmis olabilir
- asil problem hala model surface stability + overfitting + alt kuyruk dayaniksizligi

Bu nedenle siradaki dogru sira:

1. `price-only` ranker ablation
2. `sec_filings + corporate_actions` only ablation
3. `xgboost` challenger
4. ancak bundan sonra tighter ranker hardening

## 2. Veri Kullanim Politikasi

### 2.1 Su anda primer egitim verisi

Primer egitim verisi su olmali:

- **15Min OHLCV:** ana training matrisi
- **Clean core universe:** su an icin onayli 11 sembollu evren
- **Frozen snapshot replay:** ayni snapshot bundle ile apples-to-apples kiyas

Bu asamada amac:

- veri genisletmek degil,
- once model secim yuzeyini ve cross-symbol davranisi stabilize etmek.

### 2.2 Database katmanlarini ne kadar kullanacagiz

| Katman | Simdi | Sonra | Karar |
|---|---:|---:|---|
| `ohlcv_bars` 15Min | %100 | %100 | Ana training kaynagi |
| `ohlcv_bars` 1Day | %0 ilk asama | %100 ikinci asama | Multi-timeframe context icin sonra acilacak |
| `ohlcv_bars` 1Min | %0 primer training | Sinirli/ayri calisma | Simdilik ana model icin gereksiz karmasiklik |
| `macro_observations` + `macro_vintage_observations` | %0 primer branch | %100 ayri reference branch | Kaliteli ama mevcut reference switch cok toplu |
| `sec_filings` | %0 primer branch | %100 ayri reference branch | Tarihsel derinlik iyi, sonra eklenmeli |
| `news_articles` | %0 primer branch | Kontrollu ablation | Sadece son 1 yil var, primer modele erken dahil edilmemeli |
| `corporate_actions` | dolayli | dolayli | Veri temizligi/uyarlama icin onemli, alpha driver olarak simdilik degil |
| `earnings_events` | %0 | Sonra | `reported_date` bosluklari var, once timestamp kalitesi duzelmeli |
| `fundamental_snapshots` | %0 | Hayir, simdilik | Tek gun snapshot, training icin cok seyrek |
| `short_sale_volumes` | %0 | Hayir, simdilik | 2 haftalik veri production candidate icin yetersiz |
| `fails_to_deliver` | %0 | Hayir, simdilik | ~2 aylik veri production candidate icin yetersiz |

### 2.3 Onemli veri karari

Mevcut kodda `enable_reference_features` tek switch ile macro, SEC, news, fundamentals, earnings, short-sale ve FTD katmanlarini topluca aciyor. Bu yuzden:

- **Primer production-candidate branch'te `enable_reference_features=false` kalmali.**
- Reference feature branch ancak base price model stabilize olduktan sonra acilmali.
- Uzun vadede yapilacak dogru is, reference feature switch'ini **kaynak bazli** ayirmaktir.

Guncel uygulama notu:

- Source-selective reference feature policy eklenmelidir / eklendiginde kullanilmalidir.
- **Su an acilmasi onerilen kaynaklar:** `macro`, `sec_filings`, `news`, `corporate_actions`
- **Su an kapali kalmasi onerilen kaynaklar:** `fundamentals`, `earnings`, `short_sale`, `ftd`

Nedeni:

- `macro` ve `sec_filings` tarihsel olarak derin ve point-in-time kullanim icin yeterli
- `news` mevcut approved snapshot penceresini kapsiyor
- `corporate_actions` veri temizligi ve event context icin guvenli
- `fundamentals` tek-gun snapshot
- `earnings` history var ama availability timestamp kalitesi yetersiz
- `short_sale` ve `ftd` coverage production-grade degil

R3 sonrasi ek not:

- `news_articles.sentiment` su an NULL; haber katmani sadece count/recency benzeri sinyal uretiyor
- `r3` final selected feature set'inde `macro` ve `news` feature'lari secilmedi
- secilen `ref_` feature'lar fiilen `sec_filings + corporate_actions` tarafindan geldi

Bu yuzden:

- sonraki kontrollu ablation'da once `price-only` kosulacak
- sonra `sec_filings + corporate_actions` only kosulacak
- `macro + news` ancak bagimsiz fayda gostermeden ana candidate stack'e zorla eklenmeyecek

### 2.4 Feature store notu

`docs/DATABASE_INVENTORY.md` verisine gore `features` tablosu halen eksik ve agirlikli olarak tek sembol iceriyor. Bu yuzden:

- feature store'u su asamada model selection icin source-of-truth kabul etmeyin
- training matrisi DB raw bars + deterministic feature materialization ile uretilsin

## 3. Model Hiyerarsisi

### 3.1 Primary: `lightgbm_ranker`

Bunu neden primary tutuyoruz:

- Ayni timestamp icindeki sembolleri siralamak istedigimiz icin problem formuna uyuyor
- Promotion/runtime contract artik `query_percentile + timestamp query` semantigini tasiyor
- Cross-sectional feature set ile daha dogal eslesiyor

Bunu ne zaman kullanacagiz:

- research ve hardening icin ilk aday
- paper trading icin ancak promotion/replay tarafinda gecerse

### 3.2 Challenger: `xgboost`

Neden gerekli:

- Farkli bir tree familyasi
- Ranker basarisiz olursa daha klasik classifier pipeline ile fallback sansi verir
- Meta-labeling ve calibration tarafi genelde daha ongorulebilir davranir

Rolu:

- ranker'in alpha'sini gercekten yenip yenmedigini gormek
- production icin alternatif rota olusturmak

### 3.3 Sanity baseline: `elastic_net`

Neden gerekli:

- Eger karmasik modeller sadece noise optimize ediyorsa bunu en hizli baseline gosterir
- Feature leakage, threshold shaping veya overfit kaynakli sahte performansi ayiklamaya yardim eder

Rolu:

- absolute performance icin degil
- "karmasik model gercekten ek deger uretiyor mu" testi icin

### 3.4 Opsiyonel family check: `lightgbm` classifier

Bunu sadece su durumda train edecegiz:

- `xgboost` ile `lightgbm_ranker` arasinda sonuclar birbirine cok yakin ise
- LightGBM familyasi icinde ranker vs classifier farkini gormek istiyorsak

### 3.5 Simdilik train etmeyecegimiz modeller

- `tcn`
- `lstm`
- `transformer`
- production ensemble

Gerekce:

- daha yuksek debug maliyeti
- daha zor reproducibility
- promotion oncesi failure analysis zorlasiyor
- mevcut veri ve hedef problemi icin tabular tree modeller yeterince guclu

## 4. Training Siralamasi

### 4.1 Asama 0: Son tamamlanan run'lari yorumla

Ilk adim son iki kosunun ne anlattigini netlestirmektir:

- `wave1_ranker_research_20260402_r2_covtailfix_h12`
- `wave1_ranker_research_20260402_r3_refcurated_iso_h12`

Bu iki run'da ozellikle bakilacak metrikler:

- `mean_sharpe`
- `expected_edge_trained`
- `holdout_symbol_sharpe_p25`
- `pbo`
- `white_reality_check_pvalue`
- accepted outer fold sayisi

Guncel karar:

- `r3`, curated references ile tek basina devam etmeyi hakli cikarmadi
- bundan sonra source contribution'i izole etmek gerekiyor

### 4.2 Asama 1: Ranker base stabilization

Amac:

- ayni frozen snapshot uzerinde yalnizca **tek eksen** degistirerek ranker'i stabilize etmek

Onerilen sira:

1. `price-only` ranker calistir
2. `sec_filings + corporate_actions` only ranker calistir
3. Bu iki ablation ile reference katmaninin gercek katkisini olc
4. Sonra gerekirse calibration/tail/concentration ayari yap
5. Outer stability hala bozuksa search space daraltma patch'i yap

Bu blokta her run arasinda en fazla **bir ana davranis degisikligi** yapilmali.

### 4.3 Asama 2: Challenger block

Ranker ablation block'tan sonra ayni snapshot uzerinde su modeller train edilmeli:

1. `xgboost` h12 classifier
2. `elastic_net` h12 baseline

Burada amac:

- "ranker mi gercekten en iyi, yoksa sadece current stack'e iyi uyuyor" sorusunu cevaplamak

### 4.4 Asama 3: Multi-timeframe branch

Bu asamaya yalnizca asagidaki durumlarda gec:

- en az bir ranker run'i research seviyesinde kabul edilebilir hale gelirse
- en az bir challenger run'i da karsilastirma icin hazirsa

Bu asamada:

- `timeframes = [15Min, 1Day]`
- base timeframe yine `15Min`

Simdilik oneri:

- `1Min` katmani ana alpha modeline acilmasin
- intraday execution/tick calismasi olarak ayri tutulmali

### 4.5 Asama 4: Promotion candidate freeze

Bu asamada yeni bir snapshot freeze edilir ve **en iyi iki model** promotion profile ile tekrar train edilir:

1. best `lightgbm_ranker`
2. best `xgboost`

Amac:

- research kazananini promotion-grade artifact'e cevirmek
- replay ve paper icin resmi promotion package uretmek

### 4.6 Asama 5: Artifact-driven replay ve backtest

Promotion candidate icin su sira zorunlu:

1. replay manifest ile deterministic replay
2. promotion package ile artifact-driven backtest
3. farkli piyasa rejimlerinde birkac kesit backtest

Backtest sirasi:

- yakin donem trend
- sikisik yatay donem
- volatil stress donemi

### 4.7 Asama 6: Paper trading

Paper'a yalnizca su durumda gec:

- promotion gates gecmis model var
- replay tutarli
- recent-window backtest'te cross-symbol tail bozulmuyor

Ilk paper rollout:

1. 5 islem gunu `dry_run`
2. 10 islem gunu `paper`, kucuk evren ve dusuk pozisyon yogunlugu
3. Sonra tam clean-core universe

## 5. Parametre Ac/Kapa Matrisi

| Parametre / Switch | Research Smoke | Research Hardening | Promotion Candidate | Not |
|---|---|---|---|---|
| `training_profile` | `research` | `research` | `promotion` | Promotion profilinde SHAP + meta zorunlu |
| `strict_snapshot_replay` | acik | acik | acik | Karsilastirilabilirlik icin |
| `feature_groups` | `technical statistical microstructure cross_sectional` | ayni | ayni | Primer branch |
| `enable_cross_sectional` | acik | acik | acik | Ranker icin zorunlu |
| `enable_tick_microstructure_features` | acik | acik | acik | Sorun cikarirsa sadece diagnostikte kapat |
| `enable_reference_features` | `price-only` veya curated subset | source-selective | source-selective | Full all-on kullanma |
| `reference_feature_sources` | ilk test `kapali`, ikinci test `sec_filings corporate_actions` | ancak kanitlanirsa `macro news` eklenir | kanitlanan subset | `fundamentals earnings short_sale ftd` kapali |
| `timeframes` | sadece `15Min` | `15Min`, sonra `15Min+1Day` | `15Min+1Day` | Basarili adaylar icin |
| `n_trials` | 30 | 80-120 | 180+ | Promotion/live icin daha siki |
| `n_splits` | 3 | 4 | 6 | |
| `nested_outer_splits` | 2 | 3 | 5 | |
| `nested_inner_splits` | 2 | 3 | 4 | |
| `holdout_pct` | 0.15 | 0.20 | 0.20 | |
| `feature_selection_max_features` | 120-180 | 140-180 | 120-160 | Promotion'da fazla genislemeyin |
| `feature_selection_stability_iterations` | 12 | 16 | 16 | |
| `feature_selection_min_stability_support` | 0.60 | 0.60-0.65 | 0.65 | |
| `probability_calibration` ranker | kapali veya ablation | controlled ablation | sadece faydasi kanitlanirsa | Ranker icin ekstra serbestlik vermeyin |
| `probability_calibration` classifier | acik | acik | acik | Ozellikle `xgboost` challenger icin |
| `meta_labeling` | kapali | kapali | acik | Promotion profile bunu istiyor |
| `compute_shap` | kapali | kapali | acik | Promotion profile bunu istiyor |
| `dynamic_no_trade_band` | acik | acik | acik | Sadece dead-score diagnostiğinde kapat |
| `allow_feature_group_fallback` | kapali | kapali | kapali | Candidate run deterministik olmali |
| `execution_turnover_cap` | 0.60 | 0.45-0.50 | 0.35-0.45 | Paper oncesi sikilastir |
| `execution_max_symbol_entry_share` | 0.68 | 0.55 | 0.50 | Tail problemi icin onemli |
| `min_confidence_position_scale` | 0.20 | 0.30 | 0.35 | Paper icin daha konservatif |
| `objective_weight_tail_risk` | 0.35 | 0.45 gerekirse | 0.45 | Symbol-tail penalty artik buna bagli |
| `objective_weight_symbol_concentration` | 0.20 | 0.30 | 0.30 | |

## 6. Ic Go/No-Go Esikleri

Code gate'leri tek basina yeterli degil. Ic karar icin daha siki esikler kullanilmali.

### 6.1 Research -> Hardening gecisi

Minimum:

- booster canli olmali
- `expected_edge_trained = true`
- `holdout_symbol_sharpe_p25 >= -0.10`
- `pbo <= 0.45`
- accepted outer fold sayisi en az `1`

### 6.2 Hardening -> Promotion gecisi

Minimum:

- `holdout_symbol_sharpe_p25 >= 0.00`
- `pbo <= 0.30`
- `holdout_max_drawdown <= 0.25`
- `mean_risk_adjusted_score > 0`
- expected-edge holdout lift pozitif
- worst 1-2 sembolde cokme yok

### 6.3 Promotion -> Paper gecisi

Minimum:

- official validation pass
- replay deterministik
- recent-window backtest'lerde ciddi rejim kirilmasi yok
- universe quality gate paper runtime'da geciyor

## 7. Somut Run Sirasi

Onerilen sira budur:

1. **Run D0:** `price-only` ranker
2. **Run D1:** `sec_filings + corporate_actions` only ranker
3. **Karar:** `price-only` mi yoksa `sec+corp` mu daha saglam?
4. **Run E0:** secilen base stack ile `xgboost`
5. **Run E1:** secilen base stack ile `elastic_net`
6. **Ancak bundan sonra:** tighter ranker hardening
7. **En iyi ranker + 1Day timeframe**
8. **En iyi challenger + 1Day timeframe**
9. **Yeni frozen snapshot ile promotion run:** best ranker
10. **Yeni frozen snapshot ile promotion run:** best xgboost
11. **Replay + backtest**
12. **Dry-run**
13. **Paper**

Her blok arasinda karar kurali:

- Eger bir run sadece aggregate metrikte iyi ama `p25` veya `pbo` kotu ise, onu genisletme
- Once ayni blokta duzelt, sonra sonraki asamaya gec
- Eger `price-only` `p25` ve `pbo`'yu iyilestirirse, reference katmanlarini agresif kullanma
- Eger `sec+corp` `price-only`'dan iyi cikarsa, production adayi reference subset'i bu iki kaynak olacak

## 8. Hangi Komut Tiplerini Kullanacagiz

### 8.1 Ranker research

WSL clean launcher kullanilmali:

```bash
scripts/launch_wave1_wsl_clean_ranker_research_h12.sh <suffix>
```

Bu yolun avantaji:

- approved snapshot sabit
- symbols file sabit
- research profile sabit
- apples-to-apples karsilastirma kolay

### 8.1.1 Price-only ablation

```bash
scripts/launch_wave1_wsl_runD0_clean_ranker_priceonly_h12.sh
```

### 8.1.2 SEC + Corporate Actions only ablation

```bash
scripts/launch_wave1_wsl_runD1_clean_ranker_seccorp_h12.sh
```

### 8.2 XGBoost challenger

XGBoost icin de ayni snapshot ve ayni universe zorunlu tutulmali. Ama base stack secimi `price-only` ve `sec+corp` ablation'larindan sonra verilmelidir.

Ilk tercih:

- hangi stack ranker tarafinda daha dusuk `pbo` ve daha iyi `p25` uretiyorsa, `xgboost` o stack ile kosulmali

Mantik su olmali:

```bash
.venv/bin/python main.py train \
  --model xgboost \
  --name wave1_xgb_research_<suffix> \
  --training-profile research \
  --dataset-snapshot-bundle models/snapshots/<snap>/dataset_bundle.manifest.json \
  --strict-snapshot-replay \
  --symbols-file data/training/universes/wave1_clean_core11_20260402.json \
  --timeframe 15Min \
  --primary-horizon 12 \
  --feature-groups technical statistical microstructure cross_sectional
```

### 8.3 Elastic Net baseline

Mantik ayni, sadece model degisir. Baseline da ranker ablation sonucu secilen stack ile kosulmalidir:

```bash
.venv/bin/python main.py train \
  --model elastic_net \
  --name wave1_elastic_research_<suffix> \
  --training-profile research \
  --dataset-snapshot-bundle models/snapshots/<snap>/dataset_bundle.manifest.json \
  --strict-snapshot-replay \
  --symbols-file data/training/universes/wave1_clean_core11_20260402.json \
  --timeframe 15Min \
  --primary-horizon 12 \
  --feature-groups technical statistical microstructure cross_sectional
```

## 9. Backtest ve Paper Trading Onerisi

### 9.1 Backtest

Backtest daima **promotion package** ile yapilmali. Nedeni:

- signal thresholds
- expected-edge policy
- side policy
- ranker scoring contract
- cost model
- sizing policy

tek artifact icinde tasiniyor.

Bu repo icin dogru yol:

1. train
2. promotion package uret
3. promotion package ile backtest

### 9.2 Paper trading

Paper trading'de `scripts/trade.py --promotion-package ...` kullanilmali. Repo bunu destekliyor ve promotion adapter ranker icin cross-sectional panel scoring yapiyor.

Ilk paper rollout onerim:

1. `dry_run`, 5 islem gunu
2. `paper`, ilk 10 islem gunu
3. `max_total_positions <= 5`
4. clean-core universe'in sadece en stabil 5 sembolu ile basla
5. sonra 11 sembole cik

Paper surecinde izlenecek baslica tablolar:

- `signals`
- `model_predictions`
- `orders`
- `trades`
- `risk_events`
- `daily_performance`

## 10. Son Oneriler

En kritik prensipler:

- Ayni anda cok fazla knob degistirmeyin
- Once frozen snapshot uzerinde karsilastirin
- Reference feature katmanini primer candidate'a erken eklemeyin
- `lightgbm_ranker` primary olsun ama tek yol olmasin
- `xgboost` challenger ve `elastic_net` baseline her zaman olsun
- Deep model ve ensemble'i paper pozitiflesmeden acmayin

Bugunden sonraki en dogru sequence:

1. `price-only` ranker sonucunu al
2. `sec+corp only` ranker ile karsilastir
3. kazanan base stack ile `xgboost` ve `elastic_net` train et
4. sadece sonra tighter ranker hardening uygula
5. kazananlari `15Min + 1Day` ile tekrar dene
6. sonra promotion + replay + paper hattina gec
