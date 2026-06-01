"""Traced DP-SQL pipeline for the live web demo.

This drives the **real** DPMiddleware's own components (BudgetLedger, the
SemanticMatcher, the PredictiveAllocator, the DB, and `_add_noise`) in the exact
order of `DPMiddleware.execute()`, narrating every step it takes. It is the real
system, just instrumented -- no production code is modified. Every one of the six
execution modes (Exact, Naive, Workload, Semantic, Temporal, Predictive) is
covered, with the mode-specific steps (L1 template+param cache, L2 semantic
match, temporal clock tick / staleness eviction, predictive pre-cache allocation,
sequential composition) shown explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dpdb.analyzer import analyze_sensitivity, SensitivityError
from dpdb.budget import BudgetExhausted
from dpdb.config import Config
from dpdb.middleware import DPMiddleware, ExecutionMode
from dpdb.parser import ParseError, parse_query
from dpdb.template import extract_template, param_hash, template_hash

MODES = {
    "exact": ExecutionMode.EXACT,
    "naive": ExecutionMode.NAIVE_DP,
    "workload": ExecutionMode.WORKLOAD_DP,
    "semantic": ExecutionMode.SEMANTIC_DP,
    "temporal": ExecutionMode.TEMPORAL_DP,
    "predictive": ExecutionMode.PREDICTIVE_DP,
}
MODE_DESC = {
    "exact": "EXACT — no DP, true answer (baseline reference)",
    "naive": "NAIVE — fixed ε every query, no cache (budget burns fast)",
    "workload": "WORKLOAD-AWARE — fixed ε, exact-repeat cache is free",
    "semantic": "SEMANTIC — adds an L2 tree-kernel/embedding similarity cache",
    "temporal": "TEMPORAL — workload-aware + staleness τ / update rate λ",
    "predictive": "PREDICTIVE — model-driven ε_q = B/Û, cache free",
}


def _step(i, name, status, detail):
    return {"id": i, "name": name, "status": status, "detail": detail}


class DemoSession:
    def __init__(self, mode="predictive", total_budget=10.0, eps_fixed=1.0,
                 k_total=20, tau=5.0, lam=0.3):
        self.mode = mode
        self.B = float(total_budget)
        self.eps_fixed = float(eps_fixed)
        self.k_total = int(k_total)
        self.tau = float(tau)
        self.lam = float(lam)
        cfg = Config.from_yaml()
        cfg.privacy.total_epsilon = self.B
        cfg.privacy.default_query_epsilon = self.eps_fixed
        self.cfg = cfg
        self.mw = DPMiddleware(
            cfg, mode=MODES[mode],
            staleness_tolerance=(tau if mode == "temporal" else float("inf")),
            update_rate=(lam if mode == "temporal" else 0.0),
            update_invalidation_prob=(0.5 if mode == "temporal" else 0.0),
            predictive_k_total=self.k_total, predictive_warmup_fraction=0.2)
        self.queries_seen = 0
        self.unique: set[str] = set()
        self.history: list[dict] = []

    # ---- state ---------------------------------------------------------
    def _cache_size(self):
        b = self.mw.budget
        return sum(len(p) for p in b._cache.values()) if b else 0

    def _uhat(self):
        p = self.mw.predictor
        if p is None or len(p.template_hashes) < 2:
            return float(self.k_total)
        return p._predicted_total_unique()

    def _state(self):
        b = self.mw.budget
        spent = b.consumed_epsilon if b else 0.0
        u_k = len(self.unique)                       # distinct TEMPLATES (model u_k)
        # actual budget savings come from cache HITS (exact template+param repeats),
        # not from template repeats: a parametric template with new literals is a
        # paid MISS. Report the true, budget-faithful savings here.
        hits = sum(1 for h in self.history if h.get("hit"))
        st = {
            "mode": self.mode,
            "budget_total": round(self.B, 4),
            "budget_spent": round(spent, 4),
            "budget_remaining": round(self.B - spent, 4) if b else None,
            "cache_size": self._cache_size(),
            "queries_seen": self.queries_seen,
            "u_k": u_k,
            "cache_hits": hits,
            "savings_pct": round(100 * hits / self.queries_seen, 1) if self.queries_seen else 0.0,
            "uhat": round(self._uhat(), 2) if self.mode == "predictive" else None,
            "semantic_hits": b.semantic_hits if b else 0,
            "expired_evictions": (b.expired_evictions + getattr(b, "update_evictions", 0)) if b else 0,
            "logical_time": getattr(b, "_logical_time", 0) if b else 0,
            "history": self.history,
        }
        return st

    @staticmethod
    def _disp(parsed, rows):
        """Human-readable value(s) for the aggregate result column(s)."""
        if not rows:
            return "∅"
        pos = parsed.aggregates[0].position
        if parsed.group_by:
            vals = [str(r[pos]) for r in rows[:3]]
            return f"{len(rows)} groups: " + ", ".join(vals) + (" …" if len(rows) > 3 else "")
        v = rows[0][pos]
        return v

    # ---- the traced step (mirrors DPMiddleware.execute) ----------------
    def step(self, sql: str) -> dict:
        steps = []
        self.queries_seen += 1
        mw, b = self.mw, self.mw.budget

        steps.append(_step(0, "Mode", "ok", MODE_DESC[self.mode]))

        # EXACT short-circuit -------------------------------------------
        if self.mode == "exact":
            try:
                cols, rows = mw.db.execute_with_columns(sql)
            except Exception as e:
                self.queries_seen -= 1
                return {"ok": False, "error": str(e),
                        "steps": steps + [_step(1, "Execute", "error", str(e))],
                        "state": self._state()}
            parsed = None
            try:
                parsed = parse_query(sql)
            except ParseError:
                pass
            val = self._disp(parsed, rows) if parsed else (rows[0][0] if rows else "∅")
            steps.append(_step(1, "Execute (exact)", "ok",
                               f"true answer released directly, ε=0 (no privacy) → {val}"))
            self.history.append({"hit": False, "eps": 0.0, "err": 0.0, "template": "exact"})
            return {"ok": True, "steps": steps,
                    "final": {"kind": "exact", "true": val, "noisy": val, "err": 0.0, "eps": 0.0},
                    "state": self._state()}

        # (1) PARSE ------------------------------------------------------
        try:
            parsed = parse_query(sql)
        except ParseError as e:
            self.queries_seen -= 1
            return {"ok": False, "error": f"Parse error: {e}",
                    "steps": steps + [_step(1, "Parse & validate", "error", str(e))],
                    "state": self._state()}
        aggs = ", ".join(f"{a.func}({a.column or '*'})@col{a.position}"
                         for a in parsed.aggregates)
        steps.append(_step(1, "Parse & validate", "ok",
                           f"table={parsed.table} | {aggs} | "
                           f"WHERE={parsed.where_clause or '—'} | "
                           f"GROUP BY={', '.join(parsed.group_by) or '—'}  "
                           f"(HAVING / COUNT(DISTINCT) would be rejected here)"))

        # (2) PREDICTIVE pre-cache ε (real code computes this before cache)
        eps = self.eps_fixed
        if self.mode == "predictive" and mw.predictor is not None:
            warm = mw.predictor.queries_seen < mw.predictor.warmup_size
            eps = mw.predictor.next_epsilon(parsed)
            uhat = self._uhat()
            steps.append(_step(2, "Predictive allocate (pre-cache)", "ok",
                               f"{'WARMUP: ε_q=B/k' if warm else 'ACTIVE: ε_q=B/Û'} "
                               f"→ Û={uhat:.2f}, proposes ε_q={eps:.3f}"))

        # (3) TEMPLATE + PARAM hash -------------------------------------
        th = template_hash(extract_template(parsed))
        ph = param_hash(parsed)
        first = th not in self.unique
        self.unique.add(th)
        steps.append(_step(3, "Template + parameter hash", "ok",
                           f"template #{th[:8]} ({'NEW' if first else 'seen'}) | "
                           f"param #{ph[:8]}  — exact match needs BOTH"))

        # (4) CACHE LOOKUP (real ledger; ticks clock, L1 then L2) --------
        if b is not None:
            sem0, exp0, upd0 = b.semantic_hits, b.expired_evictions, getattr(b, "update_evictions", 0)
            t0 = getattr(b, "_logical_time", 0)
            cached = b.try_cache(parsed)
            t1 = getattr(b, "_logical_time", 0)
            if self.mode == "temporal":
                ev = (b.expired_evictions - exp0) + (getattr(b, "update_evictions", 0) - upd0)
                steps.append(_step(4, "Temporal clock tick", "ok",
                                   f"logical t: {t0}→{t1} | λ={self.lam} update sim | "
                                   f"τ={self.tau} staleness | evicted {ev} stale entr"
                                   f"{'y' if ev == 1 else 'ies'} this tick"))
            if cached is not None:
                is_l2 = b.semantic_hits > sem0
                steps.append(_step(5, "Cache lookup", "hit",
                                   ("L2 SEMANTIC match (tree-kernel/embedding) — "
                                    "⚠ similarity ≠ equivalence, may be a WRONG answer"
                                    if is_l2 else
                                    f"L1 EXACT match #{th[:8]}/{ph[:8]}") +
                                   " → return cached noisy answer FREE (Δε=0)"))
                val = self._disp(parsed, cached.rows)
                self.history.append({"hit": True, "eps": 0.0, "err": None,
                                     "template": th[:8], "l2": is_l2})
                return {"ok": True, "steps": steps,
                        "final": {"kind": "l2_hit" if is_l2 else "cache_hit",
                                  "true": "(cached)", "noisy": val, "err": None, "eps": 0.0},
                        "state": self._state()}
            miss_txt = ("naive mode → cache disabled, always recompute"
                        if self.mode == "naive"
                        else f"L1 MISS #{th[:8]}/{ph[:8]}" +
                        (" + L2 semantic MISS" if self.mode == "semantic" else "") +
                        " → must spend budget")
            steps.append(_step(5, "Cache lookup", "miss", miss_txt))

        # (5b) PREDICTIVE zero-budget reject ----------------------------
        if self.mode == "predictive" and eps <= 0:
            steps.append(_step(6, "Budget check", "reject",
                               "predictive allocator returned ε=0 (budget dry) → REJECT"))
            self.history.append({"hit": False, "eps": 0.0, "err": None,
                                 "template": th[:8], "rejected": True})
            return {"ok": True, "steps": steps, "final": {"kind": "rejected"},
                    "state": self._state()}

        # (6) SENSITIVITY ------------------------------------------------
        try:
            sens = analyze_sensitivity(parsed, self.cfg)
        except SensitivityError as e:
            return {"ok": False, "error": str(e),
                    "steps": steps + [_step(6, "Sensitivity", "error", str(e))],
                    "state": self._state()}
        sdesc = " | ".join(f"{s.func}: Δf={s.sensitivity:g}" for s in sens)
        steps.append(_step(6, "Sensitivity analysis", "ok",
                           f"{sdesc}  ({sens[0].notes})"))

        # (7) ALLOCATE / ledger (real composition) ----------------------
        try:
            allocated = b.allocate(parsed, eps)
        except BudgetExhausted as e:
            steps.append(_step(7, "Allocate budget", "reject",
                               f"REJECT — {e}"))
            self.history.append({"hit": False, "eps": 0.0, "err": None,
                                 "template": th[:8], "rejected": True})
            return {"ok": True, "steps": steps, "final": {"kind": "rejected"},
                    "state": self._state()}
        steps.append(_step(7, "Allocate budget (sequential composition)", "ok",
                           f"charge ε={allocated:.3f} | consumed {b.consumed_epsilon:.3f}"
                           f"/{self.B:g} | remaining {b.remaining:.3f}"))

        # (8) EXECUTE TRUE ----------------------------------------------
        cols, true_rows = mw.db.execute_with_columns(sql)
        true_disp = self._disp(parsed, true_rows)
        steps.append(_step(8, "Execute true query", "ok",
                           f"on real Adult table → true = {true_disp} (kept secret)"))

        # (9) LAPLACE MECHANISM (real _add_noise: per-position) ----------
        noisy_rows = mw._add_noise(parsed, true_rows, sens, allocated)
        eps_per = allocated / max(len(sens), 1)
        noisy_disp = self._disp(parsed, noisy_rows)
        if parsed.group_by:
            err = sum(abs(float(n[parsed.aggregates[0].position]) - float(t[parsed.aggregates[0].position]))
                      for t, n in zip(true_rows, noisy_rows)) / max(len(true_rows), 1)
        else:
            err = abs(float(noisy_rows[0][parsed.aggregates[0].position]) -
                      float(true_rows[0][parsed.aggregates[0].position]))
        steps.append(_step(9, "Laplace mechanism", "ok",
                           f"per-agg ε={eps_per:.3f}, scale Δf/ε; noised at the recorded "
                           f"SELECT position (groups: parallel composition) → "
                           f"noisy = {noisy_disp}  |err|={err:.2f}"))

        # (10) CACHE WRITE ----------------------------------------------
        b.store_result(parsed, cols, noisy_rows, allocated)
        steps.append(_step(10, "Cache write", "ok",
                           f"store #{th[:8]}/{ph[:8]} (free on exact repeat)"))

        # (11) PREDICTIVE bookkeeping -----------------------------------
        if self.mode == "predictive" and mw.predictor is not None:
            mw.predictor.note_release(parsed, allocated)
            steps.append(_step(11, "Predictive bookkeeping", "ok",
                               f"note release → Û re-estimated = {self._uhat():.2f}"))

        self.history.append({"hit": False, "eps": round(allocated, 4),
                             "err": round(err, 3), "template": th[:8]})
        return {"ok": True, "steps": steps,
                "final": {"kind": "miss", "true": true_disp, "noisy": noisy_disp,
                          "err": round(err, 3), "eps": round(allocated, 4)},
                "state": self._state()}


_TILE_A = "SELECT COUNT(*) FROM adult WHERE age >= 30"
_TILE_B = "SELECT AVG(age) FROM adult WHERE sex = ' Male'"
_TILE_C = "SELECT COUNT(*) FROM adult WHERE education = ' Bachelors'"

# Real-world use cases, each demonstrating one behaviour of the cache mechanism.
USE_CASES = [
    {
        "id": "dashboard",
        "name": {"en": "Live monitoring dashboard",
                 "tr": "Canlı izleme panosu"},
        "tag": {"en": "exact repeats → free", "tr": "birebir tekrar → bedava"},
        "desc": {
            "en": "A clinic dashboard refreshes the same KPI tiles again and again. "
                  "Identical queries are EXACT repeats, so every repeat is a free "
                  "cache hit (Δε=0) by post-processing — the budget is spent only "
                  "once per distinct tile. Best case for caching: savings approach 1.",
            "tr": "Bir klinik panosu aynı KPI kutucuklarını tekrar tekrar yeniler. "
                  "Birebir aynı sorgular EXACT tekrardır; post-processing sayesinde "
                  "her tekrar bedava önbellek isabetidir (Δε=0) — bütçe her farklı "
                  "kutucuk için yalnızca bir kez harcanır. Caching için en iyi durum."},
        "mode": "predictive",
        "queries": [_TILE_A, _TILE_B, _TILE_C, _TILE_A, _TILE_B, _TILE_A, _TILE_C, _TILE_B],
    },
    {
        "id": "parametric",
        "name": {"en": "Parametric BI report sweep",
                 "tr": "Parametrik BI rapor taraması"},
        "tag": {"en": "template ≠ exact → partial savings",
                "tr": "şablon ≠ birebir → kısmi tasarruf"},
        "desc": {
            "en": "An analyst runs one report TEMPLATE across many filter values "
                  "(age bands). All share a template, but the cache frees only EXACT "
                  "(template+parameter) repeats — so a brand-new band is a paid miss. "
                  "Watch u_k stay 1 while real savings are far below 1−1/k.",
            "tr": "Bir analist tek bir rapor ŞABLONUNU birçok filtre değeriyle "
                  "(yaş bantları) çalıştırır. Hepsi aynı şablonu paylaşır ama önbellek "
                  "yalnızca BİREBİR (şablon+parametre) tekrarları bedava yapar — yeni "
                  "bir bant ödenen bir ıskadır. u_k=1 kalırken gerçek tasarruf "
                  "1−1/k'nın çok altında."},
        "mode": "workload",
        "queries": [f"SELECT COUNT(*) FROM adult WHERE age >= {a} AND age < {a+10}"
                    for a in (20, 30, 40, 30, 20, 50, 30)],
    },
    {
        "id": "drilldown",
        "name": {"en": "Investigative drill-down",
                 "tr": "Araştırmacı drill-down"},
        "tag": {"en": "all unique → no savings", "tr": "hepsi farklı → tasarruf yok"},
        "desc": {
            "en": "An analyst narrows a question step by step, each query adding a "
                  "predicate. Every query is unique, so the cache never helps: "
                  "S(k)=0 and the budget drains like naive composition. The model "
                  "predicts this honestly — and it is the regime where reconstruction "
                  "attacks succeed, so spend ε carefully.",
            "tr": "Bir analist soruyu adım adım daraltır; her sorgu bir koşul ekler. "
                  "Her sorgu farklı olduğundan önbellek hiç işe yaramaz: S(k)=0 ve "
                  "bütçe naive gibi erir. Model bunu dürüstçe öngörür — ve reconstruction "
                  "saldırılarının başarılı olduğu rejim budur, ε'yu dikkatli harca."},
        "mode": "predictive",
        "queries": [
            "SELECT COUNT(*) FROM adult",
            "SELECT COUNT(*) FROM adult WHERE age >= 30",
            "SELECT COUNT(*) FROM adult WHERE age >= 30 AND sex = ' Male'",
            "SELECT AVG(age) FROM adult WHERE sex = ' Male'",
            "SELECT SUM(capital_gain) FROM adult WHERE age >= 50",
            "SELECT COUNT(*) FROM adult WHERE education = ' Doctorate'",
        ],
    },
    {
        "id": "temporal",
        "name": {"en": "Periodic report on changing data",
                 "tr": "Değişen veride periyodik rapor"},
        "tag": {"en": "staleness τ / updates λ → re-spend",
                "tr": "bayatlama τ / güncelleme λ → yeniden harca"},
        "desc": {
            "en": "The same report re-runs over time, but the data keeps changing. "
                  "A cached answer is only valid until it goes stale (tolerance τ) or "
                  "an update invalidates it (rate λ); then it must be re-released, "
                  "re-spending budget. Freshness has a price — watch the clock tick "
                  "and stale entries get evicted.",
            "tr": "Aynı rapor zamanla yeniden çalışır ama veri sürekli değişir. "
                  "Önbellekteki cevap yalnızca bayatlayana (tolerans τ) ya da bir "
                  "güncelleme onu geçersiz kılana (oran λ) kadar geçerlidir; sonra "
                  "yeniden yayınlanmalı, bütçe tekrar harcanmalıdır. Tazeliğin bir "
                  "bedeli var — saatin işleyişini ve bayat girdilerin tahliyesini izle."},
        "mode": "temporal",
        "queries": [_TILE_A] * 7,
    },
    {
        "id": "semantic",
        "name": {"en": "Near-duplicate queries",
                 "tr": "Neredeyse-aynı sorgular"},
        "tag": {"en": "similarity ≠ equivalence (honest negative)",
                "tr": "benzerlik ≠ denklik (dürüst negatif)"},
        "desc": {
            "en": "Analysts often write the same logical query in slightly different "
                  "ways. The L2 semantic cache TRIES to reuse a 'similar enough' answer "
                  "— but similar ASTs are not equivalent queries, so it can return a "
                  "WRONG answer. This is our honest negative: a semantic DP cache is "
                  "unsafe without a symbolic equivalence prover. (L2 fires only if the "
                  "embedding model is installed.)",
            "tr": "Analistler aynı mantıksal sorguyu çoğu zaman biraz farklı yazar. "
                  "L2 semantik önbellek 'yeterince benzer' bir cevabı yeniden kullanmayı "
                  "DENER — ama benzer AST'ler denk sorgular değildir, dolayısıyla YANLIŞ "
                  "cevap dönebilir. Bizim dürüst negatif bulgumuz bu: sembolik denklik "
                  "kanıtlayıcı olmadan semantik DP önbelleği güvenli değildir. (L2 yalnızca "
                  "gömme modeli kuruluysa tetiklenir.)"},
        "mode": "semantic",
        "queries": [
            "SELECT COUNT(*) FROM adult WHERE age >= 30",
            "SELECT COUNT(*) FROM adult WHERE age > 29",
            "SELECT COUNT(*) FROM adult WHERE age >= 30 AND age < 200",
        ],
    },
]

# backward-compatible name->queries map (kept for /api/preset)
PRESETS = {uc["name"]["en"]: uc["queries"] for uc in USE_CASES}
EXAMPLES = [
    "SELECT COUNT(*) FROM adult WHERE age >= 30",
    "SELECT AVG(age) FROM adult",
    "SELECT SUM(capital_gain) FROM adult WHERE age < 40",
    "SELECT sex, COUNT(*) FROM adult GROUP BY sex",
]
