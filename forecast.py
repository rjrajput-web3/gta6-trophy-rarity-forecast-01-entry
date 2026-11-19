"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file in the input directory and writes, for each trophy, five
quantiles (5/25/50/75/95%) of its Exophase earn rate at the case target.

Approach, per trophy, on the logit scale:
  1. Fit a straight line of logit(rate) against log(1 + hours since the first
     read) over the case history (least squares), then extrapolate that line to
     the target with the slope damped toward flat, so a short or noisy history
     cannot fling the forecast.
  2. Blend the damped trend with the trophy's own last observed rate, which is
     the strongest single signal when the history is flat or short.
  3. Shrink the blend toward the median rate of the trophy's own tier
     (platinum / gold / silver / bronze). The pull grows with the horizon, so a
     4.5-day forecast leans on what trophies of the same kind usually do, while
     a 90-minute forecast barely moves from the last read.
  4. Widen the spread with the horizon (random-walk growth), the fit residual,
     and an extra term for trophies pinned at 100%, whose rate can only fall
     once new players arrive.

Pure standard library, fully deterministic: no randomness, no network, no
clock reads at forecast time beyond the case file itself.
"""

import json
import math
import os
from datetime import datetime, timezone

LEVEL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
           0.6744897501960817, 1.6448536269514722)

# Blend and spread constants, tuned on synthetic out-of-sample holdouts.
TREND_WEIGHT = 0.8        # weight on the damped trend vs the last rate
SLOPE_DAMP = 0.9          # extra damping applied to the fitted slope
SLOPE_SHRINK_N = 2.0      # slope *= n / (n + this) with n = usable reads
TREND_WINDOW_HOURS = 24.0 # fit the trend on the most recent reads only
MAX_TREND_MOVE = 2.0      # cap |trend median - last logit| in logit units
TIER_PULL_PER_DAY = 0.02  # tier-shrink weight gained per horizon day
TIER_PULL_MAX = 0.12      # cap on the tier-shrink weight
SIGMA_BASE = 0.3          # spread floor, in logit units
SIGMA_PER_ROOT_DAY = 0.18 # spread growth with sqrt(horizon days)
SIGMA_PER_RESID = 0.60    # spread added per unit of fit residual std
SIGMA_MIN = 0.55
SIGMA_MAX = 2.2
SIGMA_CENSORED_EXTRA = 0.35  # extra spread for a trophy pinned at 100%
CENSORED_RATE = 97.0      # effective rate used when the last read is >= 99.5%
DEFAULT_RATE = 1.0        # fallback rate when a trophy has no history at all


def clamp_rate(rate):
    """Keep a percent inside the range the logit can represent."""
    if rate < 0.01:
        return 0.01
    if rate > 99.99:
        return 99.99
    return rate


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value > 20.0:
        return 99.99
    if value < -20.0:
        return 0.01
    p = 1.0 / (1.0 + math.exp(-value))
    return clamp_rate(p * 100.0)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp, tolerating a trailing Z."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_line(xs, ys):
    """Least-squares line y = a + b*x; returns (intercept, slope, resid_std)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if n < 2 or sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        resid_sq = sum((y - (intercept + slope * x)) ** 2
                       for x, y in zip(xs, ys))
        resid_std = math.sqrt(resid_sq / (n - 2))
    else:
        resid_std = 0.0
    return intercept, slope, resid_std


def forecast_case(case):
    """Forecast one case dict; returns the output object for its .out file."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    target = parse_time(case["target"]) if case.get("target") else None

    # Time base: hours since the first history read. x = log1p(hours).
    first_at = None
    if history:
        first_at = parse_time(history[0]["at"])
    if first_at is None and target is not None:
        first_at = target

    def hours_since_first(moment):
        if first_at is None:
            return 0.0
        return max(0.0, (moment - first_at).total_seconds() / 3600.0)

    target_x = math.log1p(hours_since_first(target)) if target else 0.0

    # Last observed logit per trophy, and the tier medians for shrinkage.
    last_logit = {}
    tier_logit = {}
    for index, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                series.append(float(rates[index]))
        if series:
            last_logit[trophy["id"]] = logit_pct(series[-1])
        tier_logit.setdefault(trophy.get("tier") or "bronze", []).append(
            last_logit.get(trophy["id"], logit_pct(DEFAULT_RATE)))

    tier_median = {}
    for tier, values in tier_logit.items():
        tier_median[tier] = median(values)
    global_median = median(list(last_logit.values()))
    if global_median is None:
        global_median = logit_pct(DEFAULT_RATE)

    tier_pull = min(TIER_PULL_MAX, TIER_PULL_PER_DAY * horizon_days)
    spread_root = SIGMA_PER_ROOT_DAY * math.sqrt(max(0.0, horizon_days))

    rows = []
    for index, trophy in enumerate(trophies):
        tid = trophy["id"]
        tier = trophy.get("tier") or "bronze"
        prior = tier_median.get(tier, global_median)

        points = []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if index >= len(rates) or rates[index] is None:
                continue
            rate = float(rates[index])
            last_rate = rate
            points.append((parse_time(read["at"]), rate))
        # Fit the trend on the most recent day of reads only: launch-night
        # dynamics go stale fast, and the recent slope is what extrapolates.
        if len(points) > 2:
            newest = points[-1][0].timestamp()
            points = [p for p in points
                      if p[0].timestamp() >= newest - TREND_WINDOW_HOURS * 3600.0]
        xs = [math.log1p(hours_since_first(moment)) for moment, _ in points]
        ys = [logit_pct(rate) for _, rate in points]

        if last_rate is None:
            centre = prior
            sigma = SIGMA_BASE + spread_root + 0.25
        else:
            censored = last_rate >= 99.5
            if censored:
                last_l = logit_pct(CENSORED_RATE)
            else:
                last_l = logit_pct(last_rate)
            intercept, slope, resid_std = fit_line(xs, ys)
            n = len(xs)
            damped = slope * SLOPE_DAMP * (n / (n + SLOPE_SHRINK_N))
            trend = intercept + damped * target_x
            # Never let the trend run far from what was last seen.
            trend = max(last_l - MAX_TREND_MOVE,
                        min(last_l + MAX_TREND_MOVE, trend))
            core = TREND_WEIGHT * trend + (1.0 - TREND_WEIGHT) * last_l
            centre = (1.0 - tier_pull) * core + tier_pull * prior
            sigma = SIGMA_BASE + spread_root + SIGMA_PER_RESID * resid_std
            if censored:
                sigma += SIGMA_CENSORED_EXTRA

        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        quantiles = []
        previous = None
        for z in LEVEL_Z:
            value = round(inv_logit_pct(centre + z * sigma), 4)
            value = max(0.01, min(99.99, value))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": tid, "q": quantiles})

    return {"trophies": rows}


def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"))
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    run(input_dir, output_dir)


if __name__ == "__main__":
    main()
