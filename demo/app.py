"""Flask backend for the live DP-SQL pipeline demo.

Run:  python demo/app.py     then open http://127.0.0.1:5000
The server holds one workload session in memory (single-user local demo); the
/api/reset endpoint starts a fresh ledger/cache/allocator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, render_template, request

from pipeline import DemoSession, EXAMPLES, PRESETS

app = Flask(__name__)

# single in-memory session (local demo)
SESSION = {"s": DemoSession()}


@app.route("/")
def index():
    return render_template("index.html", examples=EXAMPLES,
                           presets=list(PRESETS.keys()))


@app.route("/api/reset", methods=["POST"])
def reset():
    cfg = request.get_json(force=True) or {}
    SESSION["s"] = DemoSession(
        mode=cfg.get("mode", "predictive"),
        total_budget=float(cfg.get("total_budget", 10.0)),
        eps_fixed=float(cfg.get("eps_fixed", 1.0)),
        k_total=int(cfg.get("k_total", 20)),
        tau=float(cfg.get("tau", 5.0)),
        lam=float(cfg.get("lam", 0.3)),
    )
    return jsonify({"ok": True, "state": SESSION["s"]._state()})


@app.route("/api/step", methods=["POST"])
def step():
    sql = (request.get_json(force=True) or {}).get("sql", "").strip()
    if not sql:
        return jsonify({"ok": False, "error": "Empty query"})
    return jsonify(SESSION["s"].step(sql))


@app.route("/api/preset/<name>")
def preset(name):
    return jsonify({"ok": True, "queries": PRESETS.get(name, [])})


if __name__ == "__main__":
    print("DP-SQL demo  ->  http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
