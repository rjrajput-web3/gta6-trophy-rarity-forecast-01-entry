"""Out-of-sample benchmark for the GTA VI trophy rarity forecaster.

Scoring: the competition's pinball loss on the logit scale
  loss = mean over trophies x levels of 2*max(tau*d, (tau-1)*d),
  d = logit(y) - logit(q), rates clamped to [0.01, 99.99].
Also reports typical miss (mean |y - q50| in points) and 90% coverage.

Two benchmarks (no answer file exists anywhere, so both are reconstructed):

1. PRACTICE (p00/p01): the house's own documentation states the page was
   unchanged (638 players) through 04:00Z, then refreshed at 04:30Z to
   3,868 players with every rate but the 100% one lower. The answers are
   therefore (approximately) the launch-night earner counts divided by
   3,868, with earners grown by an unknown factor between the last read
   and the refresh. Scored across earner-growth scenarios 1.0x / 1.5x /
   2.0x / 3.0x (p01's 05:00 answer uses N=3,868 and N=4,500 variants).

2. SYNTHETIC FINAL: a structural simulator of launch week. N grows as a
   power law in launch age toward ~1M tracked players by Nov 26; each
   trophy's earners follow a saturating curve in log-time toward its
   category ceiling share of N. Three scenarios (central / fast / slow
   adoption). A fake f00 case is cut at Nov 21 11:30Z (history every
   30 min, Exophase-style 2-decimal rounding and stale stretches) and
   scored at Nov 26 00:00Z. This is the decisive benchmark: the real
   final is a 4.5-day horizon with real movement, where persistence
   should lose badly.

Competitors: the house persistence baseline (kit starter, sigma 0.5) and
the house curve baseline (OLS logit vs ln(1+hours) over the last 24h,
same spread), both re-implemented here from the competition's own
documentation.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
KIT = Path.home() / "workspace/makefun/gta6-rarity-kit/gta6-trophy-rarity-forecast-01-kit-20261119"
INPUT = KIT / "input"

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def pinball(forecast, truth):
    """forecast: {id: [q05..q95]}; truth: {id: rate}. Returns (loss, miss, cov)."""
    items, miss, inside = [], [], 0
    for tid, y in truth.items():
        qs = forecast[tid]
        ly = logit(y)
        for tau, q in zip(LEVELS, qs):
            d = ly - logit(q)
            items.append(2.0 * max(tau * d, (tau - 1.0) * d))
        miss.append(abs(y - qs[2]))
        if qs[0] <= y <= qs[4]:
            inside += 1
    return sum(items) / len(items), sum(miss) / len(miss), inside / len(truth)


# ------------------------------------------------------- house baselines

def baseline_persistence(case):
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    out = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        lg = logit(r if r is not None else 0.0)
        out[t["id"]] = [round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z]
    return out


def baseline_curve(case):
    """House curve baseline: OLS of logit rate on ln(1+hours since launch)
    over the last 24h of reads; spread = random-walk + residual combined."""
    from datetime import datetime

    def ts(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

    launch = ts("2026-11-19T00:00:00Z")
    at = ts(case["at"])
    hist = [r for r in case["history"] if (at - ts(r.get("at") or r["slot"])) <= 24 * 3600]
    if len(hist) < 3:
        return baseline_persistence(case)
    d = case["horizon_minutes"] / 1440.0
    h_target = (ts(case["target"]) - launch) / 3600.0
    x_t = math.log1p(h_target)
    out = {}
    for i, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for r in hist:
            v = r["rates"][i]
            if v is None:
                continue
            xs.append(math.log1p((ts(r.get("at") or r["slot"]) - launch) / 3600.0))
            ys.append(logit(v))
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        a = my - b * mx
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2))
        spread = math.sqrt((0.5 * math.sqrt(d)) ** 2 + resid ** 2)
        lg = a + b * x_t
        out[t["id"]] = [round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z]
    return out


def run_model(case_path, out_dir):
    import os
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = str(case_path.parent)
    env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
    # run only this case: copy it alone into a temp input dir
    tmp_in = Path(out_dir).parent / "tmp_in"
    tmp_in.mkdir(parents=True, exist_ok=True)
    for f in tmp_in.glob("*.in"):
        f.unlink()
    (tmp_in / case_path.name).write_bytes(case_path.read_bytes())
    env["MAKEFUN_INPUT_DIR"] = str(tmp_in)
    subprocess.run([sys.executable, str(WORK / "main.py")], env=env,
                   check=True, capture_output=True, text=True)
    data = json.loads((Path(out_dir) / (case_path.stem + ".out")).read_text())
    return {row["id"]: row["q"] for row in data["trophies"]}


# ------------------------------------------------------- 1. practice

def practice_truth(case, n_answer, e_mult):
    """Reconstructed answer: launch-night earner counts / N_answer."""
    last = case["history"][-1]
    n_last = last["players_tracked"]
    truth = {}
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        if r is not None and r >= 99.9:
            truth[t["id"]] = 100.0
            continue
        earners = max((r or 0.0) * n_last / 100.0, 0.0) * e_mult
        truth[t["id"]] = 100.0 * earners / n_answer
    return truth


def bench_practice():
    print("== PRACTICE (reconstructed answers: N=3868 at 04:30Z, earners x E) ==")
    for name in ("p00", "p01"):
        case = json.loads((INPUT / f"{name}.in").read_text())
        mine = run_model(INPUT / f"{name}.in", WORK / "bench_out")
        pers = baseline_persistence(case)
        curv = baseline_curve(case)
        for e_mult, n_ans in ((1.0, 3868), (1.5, 3868), (2.0, 3868), (3.0, 3868),
                              (1.5, 4500), (2.0, 4500)):
            truth = practice_truth(case, n_ans, e_mult)
            lm = pinball(mine, truth)
            lp = pinball(pers, truth)
            lc = pinball(curv, truth)
            print(f"  {name} E x{e_mult} N={n_ans}: mine loss={lm[0]:.4f} "
                  f"miss={lm[1]:.2f}pp cov={lm[2]:.2f} | persist loss={lp[0]:.4f} "
                  f"miss={lp[1]:.2f}pp cov={lp[2]:.2f} | curve loss={lc[0]:.4f}")


# ------------------------------------------------------- 2. synthetic final

# category ceilings (week-one share of tracked players, %) keyed by the same
# rules as main.classify, expressed here as (slope, ceiling) per trophy id
# via the model's own classifier, imported for consistency.
sys.path.insert(0, str(WORK))
import main as model  # noqa: E402


def simulate(scenario, seed_trophies):
    """Simulate launch week. Returns (case_dict, truth_dict)."""
    from datetime import datetime, timedelta, timezone

    launch = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    deadline = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
    target = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)

    # N(t): power law in launch age, scenario-scaled.
    n_par = {"central": (638.0, 3.1, 950_000.0),
             "fast": (638.0, 3.4, 1_600_000.0),
             "slow": (638.0, 2.7, 520_000.0)}[scenario]
    n0, a_n, n_cap = n_par

    def n_at(h):
        # power law anchored at h=10h -> ~43k (observed), capped
        n = 43000.0 * ((1.0 + h) / 11.0) ** a_n
        return min(n, n_cap)

    # adoption speed per scenario
    speed = {"central": 1.0, "fast": 1.35, "slow": 0.7}[scenario]

    def rate_at(trophy, h):
        slope, ceiling, _ = model.classify(trophy)
        # saturating adoption in log-time: S = u^k/(u^k+H^k), u=1+h
        k = 1.6 * speed
        h_half = 26.0 / speed  # hours to half the ceiling (story-ish)
        if trophy["tier"] == "platinum" or "100% Completion" in trophy["description"]:
            h_half = 220.0 / speed
        elif slope < 1.0:
            h_half = 120.0 / speed
        u = 1.0 + h
        s = (u ** k) / (u ** k + h_half ** k)
        return ceiling * s

    def read_at(when):
        h = (when - launch).total_seconds() / 3600.0
        n = n_at(h)
        rates = []
        for t in seed_trophies:
            e = rate_at(t, h) * n / 100.0
            rates.append(round(100.0 * e / n, 2))
        return {"slot": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "at": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "players_tracked": int(round(n)),
                "trophies_earned": int(round(sum(rate_at(t, h) * n / 100.0
                                                 for t in seed_trophies))),
                "rates": rates}

    # history: every 30 min from 23:50 Nov 18, with Exophase-style stale
    # stretches (the page refreshes a few times a day, not every read).
    history = []
    t = launch - timedelta(minutes=10)
    last_read = None
    step = 0
    while t < deadline:
        if step % 9 == 0:  # fresh read every ~4.5h, stale repeats between
            last_read = read_at(t)
        history.append(dict(last_read))
        t += timedelta(minutes=30)
        step += 1
    # ensure the last read is fresh at the deadline
    history[-1] = read_at(deadline - timedelta(minutes=1))

    h_t = (target - launch).total_seconds() / 3600.0
    n_t = n_at(h_t)
    truth = {}
    for tr in seed_trophies:
        truth[tr["id"]] = round(rate_at(tr, h_t), 2)

    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "f00",
            "at": "2026-11-21T11:30:00Z", "horizon_minutes": 6510,
            "target": "2026-11-26T00:00:00Z",
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19",
                     "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
            "trophies": seed_trophies, "history": history,
            "notes": "synthetic benchmark case"}
    return case, truth


def bench_final():
    print("== SYNTHETIC FINAL (fake f00 cut at Nov 21, scored at Nov 26) ==")
    seed = json.loads((INPUT / "p00.in").read_text())["trophies"]
    tmp = WORK / "bench_tmp"
    tmp.mkdir(exist_ok=True)
    for scenario in ("slow", "central", "fast"):
        case, truth = simulate(scenario, seed)
        (tmp / "f00.in").write_text(json.dumps(case))
        mine = run_model(tmp / "f00.in", WORK / "bench_out")
        pers = baseline_persistence(case)
        curv = baseline_curve(case)
        lm = pinball(mine, truth)
        lp = pinball(pers, truth)
        lc = pinball(curv, truth)
        print(f"  {scenario}: mine loss={lm[0]:.4f} miss={lm[1]:.2f}pp cov={lm[2]:.2f} "
              f"| persist loss={lp[0]:.4f} miss={lp[1]:.2f}pp cov={lp[2]:.2f} "
              f"| curve loss={lc[0]:.4f} miss={lc[1]:.2f}pp cov={lc[2]:.2f}")


if __name__ == "__main__":
    bench_practice()
    bench_final()
