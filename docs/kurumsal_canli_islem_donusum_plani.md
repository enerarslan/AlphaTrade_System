# Kurumsal Seviye Canlı Alım-Satım Dönüşüm Planı

Amaç: `quant_trading_system` yapısını "hobi/prototip" seviyesinden kurumsal (JPMorgan benzeri) operasyonel güvenilirliğe taşımak.

## 1) Kritik Boşluklar (Koddan Tespit Edilenler)

1. **Order lifecycle kalıcı değil (P0)**  
   Kanıt: `quant_trading_system/execution/order_manager.py` (`_orders` in-memory), `quant_trading_system/database/repository.py` (`OrderRepository` var ama yaşam döngüsüne bağlı değil).  
   Risk: süreç çökmesi/restart sonrası açık emir durumu kaybolur; reconciliation körleşir.

2. **Idempotency çakışması hatalı emir engelleme üretebilir (P0)**  
   Kanıt: `quant_trading_system/execution/order_manager.py` (`IdempotencyKeyManager.generate_key` fiyat/TIF/bracket alanlarını kapsamıyor).  
   Risk: farklı limit/stop emirleri yanlışlıkla duplicate sayılabilir.

3. **Kill switch ve cooldown state kalıcı değil (P0)**  
   Kanıt: `quant_trading_system/risk/limits.py` (`KillSwitchState` in-process).  
   Risk: restart sonrası trading yanlışlıkla tekrar açılır.

4. **Heartbeat servisi kodda var, runtime’da başlatılmıyor (P0)**  
   Kanıt: `quant_trading_system/monitoring/metrics.py` (`HeartbeatService`), `scripts/trade.py` (setup sırasında start yok).  
   Risk: dead-man switch / PagerDuty benzeri erken uyarı zinciri fiilen çalışmaz.

5. **Data kalite kontrolleri log-only (P0)**  
   Kanıt: `quant_trading_system/data/loader.py` (`_validate_data` anomalileri topluyor ama akışı kesmiyor).  
   Risk: bozuk OHLCV/veri boşlukları sinyal-model-execution’a sızar.

6. **Alpha ve registry akışında koruma eksik (P1)**  
   Kanıt: `quant_trading_system/alpha/alpha_base.py` (`generate_signals` doğrudan compute; `compute_all/evaluate_all` exception izolasyonu zayıf).  
   Risk: tek alpha hatası tüm batch’i düşürür.

7. **Feature cache key sürümü yetersiz (P1)**  
   Kanıt: `quant_trading_system/features/feature_pipeline.py` (`_generate_cache_key`, `_generate_version` alan seti dar).  
   Risk: `universe_data` veya pipeline config değişse bile eski cache/sürüm kullanılabilir.

8. **Model governance gate’leri zorunlu akışa gömülü değil (P1)**  
   Kanıt: `quant_trading_system/models/validation_gates.py` var; `scripts/train.py` CLI’de çağrılıyor; `quant_trading_system/models/model_manager.py` train/promotion yolunda zorunlu değil.  
   Risk: gate koşulları atlanarak model promotion yapılabilir.

9. **Drift/staleness ve risk breaker mantığında test açığı (P1)**  
   Kanıt: `quant_trading_system/models/staleness_detector.py`, `quant_trading_system/risk/limits.py` güçlü mantık içeriyor; testlerde kapsam sınırlı (`tests/unit/test_models.py`, `tests/unit/test_risk.py`).  
   Risk: canlıda regression yakalanmadan üretime çıkar.

10. **Observability üretim akışına tam bağlanmamış (P1)**  
   Kanıt: `scripts/trade.py` metrics collector alıyor; order/risk/portfolio event’lerine yeterince enstrümante değil; metrik kullanımı ağırlıkla testlerde.  
   Risk: production’da “çalışıyor mu?” sorusuna hızlı, metrik-temelli cevap zor.

11. **Config ve broker health eşleşmesi zayıf (P1)**  
   Kanıt: `quant_trading_system/monitoring/health.py` broker check client init’i ile `quant_trading_system/config/settings.py`’deki resolved config arasında tutarsızlık riski.  
   Risk: false alarm / yanlış endpoint kontrolü.

12. **Secret hijyeni kritik (P0 güvenlik)**  
   Kanıt: repo kökünde `.env` içinde gerçekçi credential alanları mevcut.  
   Risk: credential sızıntısı ve hesap ele geçirme.

---

## 2) Önceliklendirilmiş Dönüşüm Backlog’u

## P0 (İlk 2-3 Hafta) - Canlı İşlem Güvenlik Duvarı

1. **Durable order state + restart reconciliation**
   - Her lifecycle geçişini DB’ye yaz (`created/submitted/rejected/partial/filled/canceled`).
   - Startup’ta broker + DB reconciliation zorunlu olsun.
   - Başarı kriteri: process kill/restart sonrası açık emirler kayıpsız geri yükleniyor.

2. **Durable kill switch**
   - `KillSwitchState` Redis/DB persisted olsun.
   - Cooldown ve trigger reason restart sonrası korunmalı.
   - Başarı kriteri: restart sonrası trading otomatik açılmıyor.

3. **Data quality hard-gate**
   - `loader._validate_data` için kritik anomali eşiği aşılırsa ingest fail/queue quarantine.
   - Başarı kriteri: invalid OHLCV ile signal pipeline’a geçiş engelleniyor.

4. **Credential/security remediation**
   - `.env` repo dışına alınsın, mevcut anahtarlar rotasyona girsin.
   - Secret yönetimi (Vault/Secrets Manager + CI masked vars).
   - Başarı kriteri: repoda plaintext secret sıfır.

5. **Heartbeat’i gerçekten çalıştır**
   - Trade session bootstrap içinde `HeartbeatService.start()` + component health update.
   - Başarı kriteri: heartbeat kesildiğinde alarm zinciri tetikleniyor.

## P1 (3-6 Hafta) - Doğruluk ve Dayanıklılık Derinleştirme

1. **Idempotency anahtarını genişlet**
   - fiyat, stop/take-profit, TIF, strategy params dahil.
   - Başarı kriteri: farklı fiyatlı emirler false-duplicate olmuyor.

2. **Alpha execution isolation**
   - `compute_all/evaluate_all` per-alpha try/except + health flag.
   - Başarı kriteri: tek alpha hatası batch’i düşürmüyor.

3. **Feature cache/version doğruluğu**
   - `universe_data` fingerprint’i ve config’in tüm etkili alanları hash’e girsin.
   - Başarı kriteri: config/universe değişiminde cache invalidation doğru çalışıyor.

4. **Model gate zorunlu promotion**
   - `ModelManager.train_model` ve model publish yolunda gate pass zorunlu.
   - Gate sonuçları registry metadata’ya yazılsın.
   - Başarı kriteri: gate fail model production’a çıkamıyor.

5. **Observability wiring**
   - order submit/fill latency, reject rate, slippage, risk reject, drawdown metrikleri zorunlu.
   - Başarı kriteri: dashboard/Prometheus’ta canlı akış görünür.

## P2 (6-12 Hafta) - Kurumsal Operasyon Katmanı

1. **Broker abstraction + failover**
   - `AlpacaClient` bağımlılığını interface arkasına al.
   - degrade/failover prosedürü ve simülasyon testi ekle.

2. **Tam audit trail (tamper-evident)**
   - risk check fail, kill switch, manual override, model promotion olayları immutable log’a yazılsın.
   - hash-chain/imza ile değişmezlik kanıtı değerlendirilsin.

3. **Pre-trade + post-trade control framework**
   - daily turnover, concentration, exposure, liquidity threshold, hard-block politikaları.
   - limit breach sonrası otomatik aksiyon kitaplığı.

4. **Institutional test strategy**
   - fault injection: broker timeout, stale market data, DB down, partial fill storm.
   - synthetic “trading day replay” integration testi.
   - SLO bazlı test gate (MTTR, error budget, alert-to-action süresi).

---

## 3) Hemen Açılacak İş Paketleri (Uygulama Sırası)

1. `execution`: order state persistence + startup reconciliation.
2. `risk`: durable kill switch + cooldown persistence.
3. `data`: hard quality gate + quarantine akışı.
4. `monitoring`: heartbeat runtime wiring + kritik metrik enstrümantasyonu.
5. `models`: validation gate’i promotion path’e gömme.
6. `features/alpha`: cache hash düzeltme + alpha isolation.
7. `security`: secret rotation + repo hygiene + CI policy.
8. `tests`: regression + chaos/fault senaryoları.

---

## 4) Ölçülebilir Hedefler (90 Gün)

1. **Order recovery success**: restart sonrası açık emir geri kazanım oranı `%100`.
2. **Risk restart safety**: kill switch aktifteyken restart sonrası trade açılma oranı `%0`.
3. **Data gate efficacy**: kritik kalite ihlali verisinin production pipeline’a geçişi `%0`.
4. **Model governance**: production’daki tüm modeller için gate-metadata kapsama oranı `%100`.
5. **Observability completeness**: order/risk/model temel metriklerinin eksiksiz yayın oranı `%95+`.
6. **Secrets hygiene**: repoda plaintext secret tespiti `%0`.

---

## 5) Not

Bu plan, doğrudan kod tabanındaki mevcut yapı ve tespit edilen boşluklardan türetilmiştir. İstersen bir sonraki adımda bunu sprint kırılımına (issue listesi + tahmini efor + bağımlılık grafı) çevirip hemen uygulamaya başlayabilirim.
