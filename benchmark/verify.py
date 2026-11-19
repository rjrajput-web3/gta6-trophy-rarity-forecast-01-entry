"""Out-of-sample check for the GTA VI trophy rarity forecaster.

Two checks, both run locally with no network:

1. Synthetic holdouts. The kit practice histories are flat (every launch-night
   read identical), so they cannot rank forecasters. Instead we simulate three
   independent launches: a growing tracked-player base, each trophy earning
   toward its own week-one rate on its own timescale, rates rounded to 2
   decimals like the real page. A case is cut at hour 48 and scored against
   the true rates at hour 168 (a 5-day horizon, like the final). Our forecast
   must beat the persistence baseline and the fitted-curve baseline on every
   seed, measured by the competition's mean pinball loss on the logit scale.

2. Practice validity. Run the real entry program on the kit's practice cases
   p00/p01 twice and require byte-identical outputs, 55 rows each, monotone
   quantiles inside [0, 100], and files under the 65,536-byte cap.

Usage: python3 benchmark/verify.py
"""

import json
import math
import os
import random
import subprocess
import sys
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(os.path.expanduser("~"), "workspace", "makefun",
                   "gta6-rarity-kit",
                   "gta6-trophy-rarity-forecast-01-kit-20261119")

sys.path.insert(0, ROOT)
import forecast as fc

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = fc.LEVEL_Z
RELEASE = "2026-11-19T00:00:00Z"

# Week-one rate medians per tier for the synthetic launches (percent).
TIER_MEDIAN = {"platinum": 0.3, "gold": 2.0, "silver": 6.0, "bronze": 18.0}
TIER_SPREAD = {"platinum": 0.7, "gold": 0.7, "silver": 0.8, "bronze": 0.9}


def pinball_loss(rows, actual):
    """Mean pinball loss on the logit scale, exactly as the rules define it."""
    total = 0.0
    count = 0
    for row in rows:
        y = fc.logit_pct(actual[row["id"] - 1])
        for tau, q in zip(LEVELS, row["q"]):
            d = y - fc.logit_pct(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count if count else float("nan")


def persistence_rows(case):
    """House persistence baseline: last rate, random-walk spread."""
    days = float(case["horizon_minutes"]) / 1440.0
    width = 0.5 * math.sqrt(days)
    rows = []
    last = case["history"][-1]["rates"]
    for trophy in case["trophies"]:
        rate = last[trophy["id"] - 1]
        centre = fc.logit_pct(rate)
        rows.append({"id": trophy["id"], "q": [
            round(fc.inv_logit_pct(centre + z * width), 4) for z in Z]})
    return rows


def curve_rows(case):
    """House curve baseline: per-trophy logit line on log time, last 24h."""
    release = fc.parse_time(RELEASE)
    target = fc.parse_time(case["target"])
    target_x = math.log1p((target - release).total_seconds() / 3600.0)
    days = float(case["horizon_minutes"]) / 1440.0
    width = 0.5 * math.sqrt(days)
    cutoff = (fc.parse_time(case["at"]) - release).total_seconds() / 3600.0
    rows = []
    for trophy in case["trophies"]:
        idx = trophy["id"] - 1
        xs, ys = [], []
        for read in case["history"]:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            hours = (fc.parse_time(read["at"]) - release).total_seconds() / 3600.0
            if hours < cutoff - 24.0:
                continue
            xs.append(math.log1p(hours))
            ys.append(fc.logit_pct(rates[idx]))
        if len(xs) < 3:
            centre = fc.logit_pct(case["history"][-1]["rates"][idx])
            sigma = width
        else:
            intercept, slope, resid = fc.fit_line(xs, ys)
            centre = intercept + slope * target_x
            sigma = math.sqrt(width * width + resid * resid)
        rows.append({"id": trophy["id"], "q": [
            round(fc.inv_logit_pct(centre + z * sigma), 4) for z in Z]})
    return rows


def synth_case(seed):
    """Simulate one launch; return (case dict, true rates at the target)."""
    rng = random.Random(seed)
    with open(os.path.join(KIT, "input", "p00.in"), encoding="utf-8") as fh:
        template = json.load(fh)
    trophies = template["trophies"]

    week_one = []
    tau_hours = []
    for trophy in trophies:
        tier = trophy["tier"]
        rate = math.exp(rng.gauss(math.log(TIER_MEDIAN[tier]),
                                  TIER_SPREAD[tier]))
        week_one.append(min(95.0, max(0.05, rate)))
        tau_hours.append(min(400.0, max(6.0, math.exp(
            rng.gauss(math.log(36.0), 0.8)))))

    def players_at(hours):
        return int(638 * (1.0 + hours / 6.0) ** 1.5)

    def rates_at(hours):
        out = []
        earned_total = 0
        for i in range(len(trophies)):
            progressed = 1.0 - math.exp(-hours / tau_hours[i])
            out.append(round(week_one[i] * progressed, 2))
        return out

    history = []
    step = 0
    hours = 0.5
    base = fc.parse_time(RELEASE)
    while hours <= 48.0 + 1e-9:
        rates = rates_at(hours)
        players = players_at(hours)
        earned = sum(int(round(r / 100.0 * players)) for r in rates)
        stamp = base + timedelta(hours=hours)
        stamp_text = stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        history.append({
            "slot": stamp.strftime("%Y-%m-%dT%H:%M:00Z"),
            "at": stamp_text,
            "players_tracked": players,
            "trophies_earned": earned,
            "rates": rates,
        })
        hours += 0.5
        step += 1

    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "synth-%d" % seed,
        "at": "2026-11-21T00:00:00Z",
        "horizon_minutes": 7200,
        "target": "2026-11-26T00:00:00Z",
        "levels": list(LEVELS),
        "game": template["game"],
        "trophies": trophies,
        "history": history,
        "notes": "synthetic launch, seed %d" % seed,
    }
    return case, rates_at(168.0)


def check_practice():
    """Run the entry on the kit practice cases twice; check validity."""
    in_dir = os.path.join(KIT, "input")
    outs = []
    for tag in ("a", "b"):
        out_dir = os.path.join("/tmp", "gta6-verify-" + tag)
        os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = in_dir
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "forecast.py")],
            env=env, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return False, "practice run %s failed: %s" % (tag, proc.stderr[-500:])
        outs.append(out_dir)
    for case_id in ("p00", "p01"):
        first = open(os.path.join(outs[0], case_id + ".out"), "rb").read()
        second = open(os.path.join(outs[1], case_id + ".out"), "rb").read()
        if first != second:
            return False, "%s not deterministic across runs" % case_id
        if len(first) > 65536:
            return False, "%s output over the byte cap" % case_id
        data = json.loads(first)
        if set(data) != {"trophies"} or len(data["trophies"]) != 55:
            return False, "%s wrong row count" % case_id
        seen = set()
        for row in data["trophies"]:
            if set(row) != {"id", "q"} or len(row["q"]) != 5:
                return False, "%s malformed row" % case_id
            if row["id"] in seen:
                return False, "%s duplicate id" % case_id
            seen.add(row["id"])
            if any(q < 0 or q > 100 for q in row["q"]):
                return False, "%s quantile out of range" % case_id
            if any(b < a for a, b in zip(row["q"], row["q"][1:])):
                return False, "%s quantiles not monotone" % case_id
        if seen != set(range(1, 56)):
            return False, "%s id set wrong" % case_id
    return True, "practice p00/p01 valid, byte-identical across 2 runs"


def median_abs_logit_error(rows, actual):
    """Mean |logit(actual) - logit(median)|: how good the centres are."""
    total = 0.0
    for row in rows:
        total += abs(fc.logit_pct(actual[row["id"] - 1])
                     - fc.logit_pct(row["q"][2]))
    return total / len(rows)


def main():
    print("seed   ours   persistence   curve   (median |logit err| ours/pers/curve)")
    ours_all, pers_all, curve_all = [], [], []
    for seed in (11, 23, 37):
        case, actual = synth_case(seed)
        ours_rows = fc.forecast_case(case)["trophies"]
        pers_rows = persistence_rows(case)
        curv_rows = curve_rows(case)
        ours = pinball_loss(ours_rows, actual)
        pers = pinball_loss(pers_rows, actual)
        curv = pinball_loss(curv_rows, actual)
        ours_all.append(ours)
        pers_all.append(pers)
        curve_all.append(curv)
        print("%-6d %.4f  %.4f       %.4f   %.3f / %.3f / %.3f" % (
            seed, ours, pers, curv,
            median_abs_logit_error(ours_rows, actual),
            median_abs_logit_error(pers_rows, actual),
            median_abs_logit_error(curv_rows, actual)))
    mean = lambda xs: sum(xs) / len(xs)
    print("mean   %.4f  %.4f       %.4f" % (
        mean(ours_all), mean(pers_all), mean(curve_all)))
    ok = all(o < p and o < c for o, p, c in
             zip(ours_all, pers_all, curve_all))
    print("beats both baselines on every seed:", ok)
    valid, message = check_practice()
    print("practice:", message)
    if not ok or not valid:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
