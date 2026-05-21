# CMP653 Proje Ozeti — Differentially Private SQL Middleware

**Yazar:** Refik Can Oztas (N25142279) | **Hacettepe Universitesi**
**Proje basligi:** *Tekrarli Aggregate SQL Workload'lari icin Privacy-Budget Tuketimi Uzerine Bir Istatistiksel Model*

---

## Problem

Aggregate SQL sorgulari (COUNT, SUM, AVG) sadece ozet istatistik dondurduklerinden "guvenli" sanilirlar. **Bu yanlistir.** Iki ust uste binen COUNT sorgusu, differencing attack ile bir bireyi izole edebilir (klasik HIV ornegi); Dinur-Nissim 2003 yeterince cok aggregate cevabin tum veritabani rekonstruksiyonuna izin verdigini ispatladi. Differential Privacy (DP) Laplace gurultusuyle bu saldirilara karsi koruma sunar, ama gercek bir DP-SQL sistemi her sorguya ayri ayri butce harcamadan **workload-seviyesinde** butceyi nasil yonetir? Bu projenin sordugu soru budur.

## Milestone'da Yaptiklarimiz (Onceki Hali)

PostgreSQL/DuckDB ustune Python'da yazilmis bir middleware:
- SQL parser/validator (COUNT/SUM/AVG + GROUP BY + WHERE)
- Sensitivity analyzer (clipping bound'lar config'den)
- Laplace mekanizmasi
- **Workload-aware budget ledger** — ayni sorgu tekrar gelirse cache'den cevap, ek butce harcamadan (post-processing property)
- 28 unit test, TPC-H SF=1 (6M satir lineitem) + UCI Adult datasetleri
- 4 workload ailesi (W1 repetitive, W2 parametric, W3 diverse, W4 drilldown)

## Hocanin Elestirileri (Yeni Hali Niye Yazdik)

Annotated PDF'te hocanin yorumlari:
1. **"What is the use of caching in DP?"** — caching'i ana katki olarak gosterdik ama DP'nin post-processing property'sinin trivial sonucu
2. **"Research question not properly explored"** — soruya net cevap yok
3. **"Algorithm 1 trivial!"** — pseudocode cok basit
4. **"1 − 1/k ?"** — bu formul nereden geliyor?
5. **"AVG — why?"** — neden SUM/COUNT bolerek implement ettin?
6. **"Caching is already working like that"** — bu zaten DP'nin bilinen ozelligi, katki degil
7. (Sayfa sonunda) **"Assumptions on exact-repeat reuse very strong. Budget and temporality should be considered together. Leakage and savings models very limited. Propose statistical model."**

Yani hoca: *Caching'i katki olmaktan cikar, **istatistiksel bir model** ile budget tuketimini ve utility'yi onceden tahmin et. Temporal etkileri (data update'leri, cache stale'ligi) da modele kat. Leakage deneylerini "planli" degil, **calistirilmis** olarak goster.*

## Major Revizyon — Hocanin Yorumlarini Adresleyen Yeni Calismalar

### R1: Katkinin Yeniden Cerceveleresi
Caching artik **mekanizma**, ana katki **kapali-form analitik model**. Eski research question (caching helps mi?) yeni RQ'nun (model gercek butceyi tahmin edebiliyor mu?) ozel hali.

### R2: Analitik Model (en kritik ekleme)
Workload'un her sorgusu, m template'li bir dagilim {p_i}'den i.i.d. cekilir. Su sonuclar turetildi:
- **Proposition 1:** `E[u_k] = Σ_i [1 − (1−p_i)^k]` — beklenen tekil sorgu sayisi
- **Proposition 2:** `E[ε_workload-aware(k)] = ε_q × E[u_k]`, savings ratio `S(k) = 1 − E[u_k]/k`
- **Limit A** (p_1=1, mukemmel tekrar): `S(k) = 1 − 1/k` → hocanin question-mark'ledigi toy formul aslinda modelin **kose durumu**!
- **Limit B** (uniform, m→∞): `S(k) → 0` → naive composition geri kazaniliyor
- **Proposition 3:** Zipf(α) parametrik durum
- **Proposition 4:** McDiarmid bounded-differences ile concentration bound
- **Proposition 5:** Sabit toplam butce altinda utility tahmini

Kod: `src/dpdb/model.py`, derivation: `report/R2_model_sketch.md`, 15 unit test ile sinaniyor.

### R3: Temporal Extension
**Proposition 6:** `E[ε_temporal(T)] = ε_q × E[u_total] × N(T,τ,λ,q)` — staleness tolerance `τ` ve update rate `λ` ile budget zaman ekseninde nasil degisir? Yeni `TEMPORAL_DP` execution mode, Algorithm 1 artik sadece "parse, cache, return" degil; update event'leri simüle ediyor, cache yaslarini takip ediyor.

### R4: Leakage Deneyleri (Calistirildi)
- **Membership Inference Attack:** Empirical AUC, teorik `e^ε/(1+e^ε)` bound'una ≤%1 hatayla esit.
- **Reconstruction (differencing-style drilldown):** Hata `~2/ε` olcekte, beklenen Laplace standart sapmasi ile uyumlu.

### R5: Inline Yorumlarin Tek Tek Cevaplari
- AVG: SUM/COUNT bolme tercihi gerekceli (alternatif `n`'yi sizdırır), `2ε_q`/group bilingli durustluk
- `1 − 1/k`: Proposition 2 Limit A olarak konumlandi
- Algorithm 1: temporal regime + AVG cost ile genisletildi

### R6: Benchmark Campaign
720 trial (6 alpha × 4 k × 30 trial): model `E[u_k]` empirical ortalamayi **22/24 hucrede %3'ten az hatayla** tahmin ediyor.

## AI/NLP Bonus — Semantic L2 Cache

Tree Kernel (Collins-Duffy 2001) + AST Embedding (CodeBERT/GraphCodeBERT yaklasimi, sentence-transformers ile) ekledim. Honest negative result: %90 budget tasarrufu sagliyor ama alpha-equivalent OLMAYAN query'ler icin yanlis cevap donduruyor. Bu trade-off paper'da net olarak gosteriliyor — sadece pozitif bulguya iliskin yayin yapmak yerine.

## Sayilar (Headline)

- Kod: ~1.4 KLOC Python, **57 unit test geciyor**
- Datasetler: TPC-H SF=1 (6M lineitem) + UCI Adult (48,842 satir)
- Model dogruluk: 22/24 hucrede <%3 hata
- MIA AUC vs teorik: ≤%1 hata
- Reconstruction error scaling: `~2/ε` ile uyumlu
- W1 repetitive workload: 10x butce tasarrufu, 20/20 sorgu yanitlandi
- Github: https://github.com/canoztas/cmp653-project (private)

## Cikti Dosyalari

- `report/final_report.tex` — 9 bolum, 18 referans, ACM SIGMOD formati
- `report/response_to_reviewer.md` — her hocanin yorumuna karsi cozum mapping'i
- `report/R2_model_sketch.md` — modelin matematik tureti
- `results/` — tum deney CSV'leri + 12 figure (PDF + PNG)
- `experiments/{model,leakage,temporal,semantic}_validation.py` — yeniden uretilebilir deney scriptleri

## Sonuc

Milestone'da hocanin "trivial" buldugu caching mekanizmasi artik **analitik bir modelin kose durumu**. Model, gercek butce tuketimini ve utility'yi onceden tahmin ediyor; temporal extension ile budget-zaman bagintisi formalize; leakage deneyleri yapilmis (sadece "planli" degil). Paper artik bir mechanism instantiation degil, bir **predictive systems study**.
