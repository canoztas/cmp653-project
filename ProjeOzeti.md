# CMP653 Proje Ozeti — Differentially Private SQL Middleware

**Yazar:** Refik Can Oztas (N25142279) | **Hacettepe Universitesi**
**Proje basligi:** *Tekrarli Aggregate SQL Workload'lari icin Privacy-Budget Tuketimi Uzerine Bir Istatistiksel Model*
**Repository:** https://github.com/canoztas/cmp653-project (private, commit `4937bbf`)

---

## Problem

Aggregate SQL sorgulari (COUNT, SUM, AVG) sadece ozet istatistik dondurduklerinden "guvenli" sanilirlar. **Bu yanlistir.** Iki ust uste binen COUNT sorgusu, differencing attack ile bir bireyi izole edebilir; Dinur-Nissim 2003 yeterince cok aggregate cevabin tum veritabani rekonstruksiyonuna izin verdigini ispatladi. Differential Privacy (DP) Laplace gurultusuyle koruma sunar, ama gercek bir DP-SQL sistemi her sorguya ayri ayri butce harcamadan **workload-seviyesinde** butceyi nasil yonetir?

## Milestone'da Yaptiklarimiz

PostgreSQL/DuckDB ustune Python middleware:
- SQL parser/validator (COUNT/SUM/AVG + GROUP BY + WHERE)
- Sensitivity analyzer
- Laplace mekanizmasi
- Workload-aware budget ledger (exact-repeat caching, post-processing property)
- 28 unit test, TPC-H SF=1 (6M satir lineitem) + UCI Adult dataset
- 4 workload ailesi (W1 repetitive, W2 parametric, W3 diverse, W4 drilldown)

## Hocanin Elestirileri

1. **"What is the use of caching in DP?"** — caching DP'nin post-processing property'sinin trivial sonucu, katki degil
2. **"Research question not properly explored"**
3. **"Algorithm 1 trivial!"**
4. **"1 − 1/k ?"** — bu formul nereden geliyor?
5. **"AVG — why?"** — neden SUM/COUNT bolerek?
6. **"Caching is already working like that"** — DP'nin bilinen ozelligi
7. **"Assumptions on exact-repeat reuse very strong. Budget and temporality should be considered together. Leakage and savings models very limited. Propose statistical model."**

## Major Revizyon — Hocanin Yorumlarini Adresleyen Calismalar

### R1: Katkinin Yeniden Cerceveleresi
Caching artik **mekanizma**, ana katki **kapali-form analitik model**. Eski research question (caching helps mi?) yeni RQ'nun (model gercek butceyi tahmin edebiliyor mu?) ozel hali.

### R2: Analitik Model (en kritik ekleme)
Workload, m template'li bir dagilim {p_i}'den i.i.d. cekiliyor. 5 proposition turetildi:
- **P1:** `E[u_k] = Σᵢ [1 − (1−pᵢ)^k]` — beklenen tekil sorgu sayisi
- **P2:** Budget savings `S(k) = 1 − E[u_k]/k`
- **P3:** Zipf(α) parametrik durum
- **P4:** McDiarmid bounded-differences ile concentration
- **P5:** Sabit toplam butce altinda utility tahmini

**Iki limit dogrulandi:**
- **Limit A** (p₁=1, mukemmel tekrar): `S(k) = 1 − 1/k` → hocanin question-mark'ledigi toy formul aslinda modelin **kose durumu**
- **Limit B** (uniform, m→∞): `S(k) → 0` → naive composition geri kazaniliyor

### R3: Temporal Extension
**P6:** `E[ε_temporal(T)] = ε_q · E[u_total] · N(T,τ,λ,q)` — staleness tolerance τ ve update rate λ ile budget zamanla nasil degisir? Yeni `TEMPORAL_DP` execution mode.

### R4: Leakage Deneyleri (Calistirildi)
- **Single-query MIA:** Empirical AUC, teorik `e^ε/(1+e^ε)` bound'una ≤%1 hatayla
- **Reconstruction (drilldown):** Hata `~2/ε` olcekte, beklenen Laplace standart sapmasiyla uyumlu
- **Shadow-model MIA across W1-W4** (32 cell × 60 shadow run): AUC 0.14-0.58 (cumulative-ε bound 1.0'dan COK dusuk). Honest finding: cumulative budget bound gerc̆ek attack icin loose

### R5: Inline Yorumlar Tek Tek
- **AVG (honest):** Mevcut implementasyon AVG'ye dogrudan Laplace uyguluyor (sensitivity = column bound, worst-case group=1). Decomposed SUM/COUNT yaklasimi §11 Future Work'te. Paper ve kod artik tutarli.
- `1 − 1/k`: Proposition 2 Limit A olarak konumlandi
- Algorithm 1: temporal regime ile zenginlestirildi

### R6: Benchmark Campaign — 4,155 core trial + 3,950 follow-up trial

| Sweep | Trial | Sonuc |
|-------|-------|-------|
| Main grid (α × k) | 720 | 21/24 hucre <3% hata |
| Extended α (10'a kadar) | 240 | Tum hucreler <2% |
| Epsilon sweep | 480 | Model ε-bagimsiz dogrulandi |
| Large k (500'e kadar) | 75 | m=7'de saturate |
| SF=1 vs SF=10 (60M satir) | 480 | Dataset-bagimsiz, 0.03 unit fark |
| Full benchmark (6 workload × 4 ε × 3 mod) | 2160 | W1 100x, W4 1x (model dogru) |

---

## AI / NLP Eklemesi (Section 8 — Semantic L2 Cache)

**Tree Kernel + AST Embedding** ile semantic similarity cache layer'i ekledim:

- **Tree Kernel (Collins-Duffy 2001):** Iki SQL AST'sinin ortak subtree sayisi. Deterministik, egitim yok, structural equivalence yakaliyor.
- **AST Embedding (sentence-transformers, all-MiniLM-L6-v2):** Canonical AST string → 384-dim dense vektör. CodeBERT/GraphCodeBERT ruhunda, kucuk off-the-shelf model.

Iki score yuksek esikleri (K_norm ≥ 0.95 AND cosine ≥ 0.98) gectiginde cache hit oluyor.

**Honest negative result:** %90 budget tasarrufu ama alpha-equivalent OLMAYAN query'ler icin yanlis cevap donduruyor. Semantic similarity ≠ formal equivalence. Bu, "AI ile DP cache'i akillilastiralim" diyenler icin somut bir uyari.

---

## Predictive Budget Allocator (Section 10 — Yeni Mekanizma)

Brief'in onerdigi "yeni mechanism" — modelin pratik kullanimi:

**Mekanizma:**
1. Online template-frequency dagilimi tut
2. Warmup'tan sonra empirical p̂'ya P1 formulu uygula: Û = E[u_k_total]
3. Per-query budget: `ε_q* = B / max(Û, 1)`

**Bu NEDEN yeni:** PINQ, PrivateSQL, Chorus, DOP-SQL — hepsinde ε_q sabit, analyst tarafindan offline secilir. Bu, **DP-SQL'da modelden surekli ε_q ureten ilk mekanizma**. Mathematical justification: Proposition 5 corollary.

**Sonuc:** MAE %4-17 daha dusuk naive'den, ayni B=10 toplam butce ile.

---

## Section 11 — Future Work (rapor §11'de 8 madde; başlıcaları)

1. **Cache re-release on budget surplus** — predictive'in en buyuk eksigi
2. **Rényi DP / zCDP composition** — daha siki budget bound
3. **Burstiness modelling** — i.i.d. yerine renewal process
4. **Real workload traces** — Snowflake/Redshift anonymized logs
5. **Joint leakage bound** — model'i cumulative-eps yerine kullan
6. **Equivalence-checked semantic cache** — similarity + symbolic prover

---

## Sayilar

| | Sayi |
|---|---|
| Core campaign trial (§7) | **4,155** |
| Follow-up trial (§8-§10) | **~3,950** |
| Aggregated result row | 5,613 |
| Toplam query | ~150,000 |
| Unit test | **73/73 passing** |
| Figure (PDF + PNG) | 24 |
| Workload | 6 |
| Execution mode | 6 (Exact, Naive, Workload, Semantic, Temporal, **Predictive**) |
| Dataset | TPC-H SF=1 (6M), SF=10 (60M), UCI Adult (48K) |
| Code | ~1.4 KLOC Python (core middleware) |

---

## Hocanin Brief'i — Tam Audit

| # | Madde | Status |
|---|-------|--------|
| R1 | Reframe contribution | ✅ |
| R2 | Statistical model | ✅ 5 prop + temporal extension |
| R3 | Temporal coupling | ✅ Algorithm 1 + 30-trial validation |
| R4 | Leakage analysis | ✅ MIA + reconstruction + shadow-MIA across W1-W4 + cross-check |
| R5 | Inline comments | ✅ Her birine cevap |
| R6 | Benchmark campaign | ✅ SF=1+SF=10, ε sweep, 30 trial/cell |
| Reproducibility | ✅ figure-to-script tablo |
| Paper | ✅ final_report.tex (12 section, 9 embed figure) |
| Response-to-reviewer | ✅ response_to_reviewer.md |
| **BONUS** Predictive allocator | ✅ Yeni mekanizma + experiment |
| **BONUS** Future Work | ✅ 6 madde |

---

## Sonuc

Milestone'da hocanin "trivial" buldugu caching mekanizmasi artik **analitik bir modelin kose durumu**. Model gercek butce tuketimini ve utility'yi onceden tahmin ediyor; temporal extension ile budget-zaman bagintisi formalize; leakage deneyleri yapilmis (sadece "planli" degil); ve YENI bir mekanizma (predictive allocator) modelin pratikteki kullanimini gosteriyor. Paper artik bir mechanism instantiation degil, bir **predictive systems study + actionable mechanism**.

**Bonus AI/AST kismi:** Tree Kernel + sentence-transformer ile semantic cache layer. Honest negative result olarak literature katki.

**Future work:** 6 somut yol — birinde devam edip workshop paper'a (PrivacyDB, TaPP) cevirilebilir.
