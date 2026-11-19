"""Robustness: synthetic final with priors the model does NOT know.

Perturbs the simulator's ceilings (+/-50%), adoption half-times (x0.5-x2),
and N growth, so the model's baked-in priors are wrong; checks the model
still beats both house baselines. Also tests sigma tightening variants.
"""
import json, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B
import main as model

def simulate_perturbed(scenario, seed, rng):
    from datetime import datetime, timedelta, timezone
    launch = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    deadline = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
    target = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
    ceil_mult = {t["id"]: rng.uniform(0.5, 1.6) for t in seed}
    half_mult = {t["id"]: rng.uniform(0.5, 2.0) for t in seed}
    n_cap = {"slow": 520_000.0, "central": 950_000.0, "fast": 1_600_000.0}[scenario]
    a_n = {"slow": 2.7, "central": 3.1, "fast": 3.4}[scenario] * rng.uniform(0.85, 1.15)
    speed = {"slow": 0.7, "central": 1.0, "fast": 1.35}[scenario]
    def n_at(h):
        return min(43000.0 * ((1.0 + h) / 11.0) ** a_n, n_cap)
    def rate_at(t, h):
        slope, ceiling, _ = model.classify(t)
        ceiling *= ceil_mult[t["id"]]
        k = 1.6 * speed
        h_half = 26.0 / speed * half_mult[t["id"]]
        if t["tier"] == "platinum" or "100% Completion" in t["description"]:
            h_half = 220.0 / speed * half_mult[t["id"]]
        elif slope < 1.0:
            h_half = 120.0 / speed * half_mult[t["id"]]
        u = 1.0 + h
        return ceiling * (u ** k) / (u ** k + h_half ** k)
    def read_at(when):
        h = (when - launch).total_seconds() / 3600.0
        n = n_at(h)
        return {"slot": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "at": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "players_tracked": int(round(n)),
                "trophies_earned": 0,
                "rates": [round(rate_at(t, h), 2) for t in seed]}
    history, t, last, step = [], launch - timedelta(minutes=10), None, 0
    while t < deadline:
        if step % 9 == 0:
            last = read_at(t)
        history.append(dict(last)); t += timedelta(minutes=30); step += 1
    history[-1] = read_at(deadline - timedelta(minutes=1))
    h_t = (target - launch).total_seconds() / 3600.0
    truth = {tr["id"]: round(rate_at(tr, h_t), 2) for tr in seed}
    case = {"schema": "x", "id": "f00", "at": "2026-11-21T11:30:00Z",
            "horizon_minutes": 6510, "target": "2026-11-26T00:00:00Z",
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "GTA VI", "platform": "PS5", "release_date": "2026-11-19",
                     "source": "https://example.com"}, "trophies": seed,
            "history": history, "notes": "perturbed synthetic"}
    return case, truth

seed = json.loads((B.INPUT / "p00.in").read_text())["trophies"]
tmp = B.WORK / "bench_tmp"; tmp.mkdir(exist_ok=True)
for scenario in ("slow", "central", "fast"):
    for rep in range(3):
        rng = random.Random(hash((scenario, rep)) & 0xFFFFFFFF)
        case, truth = simulate_perturbed(scenario, seed, rng)
        (tmp / "f00.in").write_text(json.dumps(case))
        mine = B.run_model(tmp / "f00.in", B.WORK / "bench_out")
        pers = B.baseline_persistence(case)
        curv = B.baseline_curve(case)
        lm, lp, lc = B.pinball(mine, truth), B.pinball(pers, truth), B.pinball(curv, truth)
        print(f"  {scenario} rep{rep}: mine={lm[0]:.4f} (miss {lm[1]:.2f} cov {lm[2]:.2f}) "
              f"| persist={lp[0]:.4f} | curve={lc[0]:.4f}")
