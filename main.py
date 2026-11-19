"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case (<id>.in: the house's Exophase reads of the GTA VI PS5 trophy
page up to the case checkpoint) and writes <id>.out: five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's earn rate (percent of the
players Exophase tracks) at the case target. Scored by pinball loss on the
logit scale. Pure standard library, fully deterministic, no network, no
randomness, no wall-clock use: two runs on the same case write identical bytes.

Model: earners / players-tracked decomposition, in logit space.
---------------------------------------------------------------
A trophy's rate is E/N: E earners over N players tracked. The two move for
different reasons, so they are forecast separately and divided:

* N (players tracked) grows explosively at launch and then flattens.
  Exophase's page is refreshed on its own schedule, not continuously: the
  house saw it unchanged (638 players, identical rates) from 2026-11-18
  23:48Z through 2026-11-19 04:00Z, then refreshed at 04:30Z to 3,868
  players with every rate but the 100% one lower -- a dilution event.
  By ~10:04Z on launch day the page showed rates consistent with roughly
  43,000 players tracked. Reads between refreshes are stale repeats, so
  consecutive identical reads are collapsed to one effective observation
  before any fit is done; fitting raw reads would overweight stale pages.

* E (earners per trophy) grows as players progress: story trophies fast,
  grind/collection trophies slow, the platinum and 100% gold slowest.
  Each trophy's log-earners are fit against log(1 + hours since launch)
  over the effective observations (a power law in launch age), with the
  slope shrunk toward a category prior taken from the trophy's tier and
  description, and the forecast is capped at the category's week-one
  ceiling share of the forecast N.

* When the history holds too few distinct observations to fit anything
  (the practice cases: every read identical), the program falls back to a
  refresh model: the next refresh is expected to multiply N several-fold
  while E grows only modestly, so rates fall, with a wide spread. A
  persistence forecast with a narrow band is exactly what the house
  baseline does, and it scores about 1.51 on the practice for that reason.

* Trophy 2 ("Just Like Paradise") reads 100.00% in every read, before and
  after the 04:30 refresh and still on the live page at ~10:04Z, while
  Chapter 1 sits near 0.5%: no real trophy can be held by 100% of tracked
  players while the first chapter is under 1%. It is treated as a stuck
  display value: a mixture, most mass near 100% (it has survived every
  refresh so far) and the rest on the story-trophy trajectory in case it
  corrects. Forecasting 99.99% flat would be catastrophic if it corrects;
  forecasting a story rate would be catastrophic if it does not.

* Spread: per-trophy logit sigma from fit residuals plus a diffusion term
  growing with the horizon plus a regime term (wide when the history is
  degenerate, tighter when it is rich). Quantiles are the median shifted
  by the normal quantiles times sigma, rounded to 4 decimals half-up,
  clamped to [0.01, 99.99], non-decreasing by construction.

Category priors (slopes and week-one ceilings) are calibrated on public
long-run GTA V trophy data (PSN-tracked: first story trophy ~73% of
players long-run, final mission ~17%, platinum ~0.1%, gold medals ~0.1%)
and on the launch-day Exophase reads above. They are shrinkage targets
only: with a rich history the fits dominate.
"""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

LAUNCH = "2026-11-19T00:00:00Z"


# ---------------------------------------------------------------- utilities

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
    """Round to 4 decimals, half up (matches the house baseline's JS rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC (with trailing Z) -> epoch seconds, stdlib only."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


LAUNCH_TS = parse_time(LAUNCH)


def hours_since_launch(ts):
    return max(0.0, (ts - LAUNCH_TS) / 3600.0)


def ols(xs, ys):
    """Least squares y = a + b x. Returns (a, b, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return a, b, math.sqrt(var)


# ------------------------------------------------------- trophy categories

def classify(trophy):
    """Category of a trophy from its tier, name and description.

    Returns (slope_prior, ceiling_pct, spread_extra):
      slope_prior -- prior power-law slope of log earners vs log(1+hours)
      ceiling_pct -- week-one ceiling share of tracked players (percent)
      spread_extra -- extra logit sigma for this kind of trophy
    """
    name = trophy["name"].lower()
    desc = trophy["description"].lower()
    tier = trophy["tier"]
    text = name + " " + desc

    if tier == "platinum":
        return (0.5, 0.3, 0.45)
    if "100% completion" in desc:
        return (0.6, 0.5, 0.45)
    if "complete chapter 1" in desc:
        return (2.2, 55.0, 0.15)
    if "complete chapter 2" in desc:
        return (2.1, 42.0, 0.15)
    if "complete chapter 3" in desc:
        return (2.0, 33.0, 0.15)
    if "complete chapter 4" in desc:
        return (1.9, 26.0, 0.15)
    if "complete the final mission" in desc and "for " not in desc:
        return (1.8, 15.0, 0.20)
    if "complete the final mission for" in desc:
        return (1.6, 11.0, 0.25)
    if "complete " in desc and ("mission" in desc or "heist" in desc
                                or "takedown" in desc or "operation" in desc):
        return (1.7, 13.0, 0.20)
    if "gone fishing" in desc or "complete gone fishing" in desc:
        return (2.0, 45.0, 0.25)  # trophy 2's correction regime
    if "stunt" in text or "gold medal" in text:
        return (0.9, 1.2, 0.40)
    if "collectible card" in text or "waink it at 100" in text:
        return (0.8, 1.8, 0.40)
    if "meditate" in text or "50 challenges" in text:
        return (0.9, 3.5, 0.35)
    if "discover 100 vehicles" in text:
        return (1.1, 7.0, 0.30)
    if "try each type of activity" in text:
        return (1.9, 28.0, 0.20)
    if "arcade game" in text or "fishing challenge" in text:
        return (1.9, 26.0, 0.20)
    if "win" in text and "race" in text:
        return (1.4, 10.0, 0.25)
    if "rob " in text or "bank vault" in text or "armored truck" in text:
        return (1.4, 9.0, 0.25)
    if "explore all of leonida" in text:
        return (1.3, 9.0, 0.25)
    if "strangers" in text:
        return (1.2, 7.0, 0.30)
    if "spend $" in text or "earn $" in text or "purchase all" in text \
            or "register 10 vehicles" in text:
        return (1.3, 8.0, 0.25)
    # default bronze: ordinary open-world / side content
    return (1.4, 9.0, 0.25)


# ------------------------------------------------------------------- model

def effective_observations(case):
    """Collapse consecutive identical reads to one observation each.

    Returns a list of (hours_since_launch, players_tracked, rates) with
    at least the last read kept even when everything is identical.
    """
    obs = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        ts = parse_time(read.get("at") or read["slot"])
        obs.append((hours_since_launch(ts), read["players_tracked"], read["rates"]))
    return obs


def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_ts = parse_time(case["target"])
    h_target = hours_since_launch(target_ts)
    x_target = math.log1p(h_target)

    obs = effective_observations(case)
    n_distinct = len(obs)
    last_h, last_n, last_rates = obs[-1]
    x_last = math.log1p(last_h)

    # --- N (players tracked) fit: log N vs log(1+hours), recent window only.
    # With too few distinct observations there is nothing to fit: the
    # refresh model below takes over.
    n_fit = None
    if n_distinct >= 3:
        recent = obs[-25:] if len(obs) > 25 else obs
        xs = [math.log1p(h) for h, _, _ in recent]
        ys = [math.log(max(1, n)) for _, n, _ in recent]
        a, b, resid = ols(xs, ys)
        # Growth flattens after launch: shrink the recent slope toward 0.8
        # and never extrapolate a shrinking player base.
        w = min(1.0, len(recent) / 12.0)
        b = w * min(max(b, 0.0), 2.0) + (1.0 - w) * 0.8
        n_fit = (a, b, resid)
    if n_fit is not None:
        a, b, _ = n_fit
        log_n_target = a + b * x_target
        n_target = math.exp(log_n_target)
        # Sanity: N never falls, and never grows more than 40x past the
        # last read over one forecast (launch-week growth is fast, not magic).
        n_target = min(max(n_target, last_n), last_n * 40.0)
    else:
        n_target = None

    rows = []
    for i, trophy in enumerate(trophies):
        slope_prior, ceiling, spread_extra = classify(trophy)
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        earners_last = max(last_rate * last_n / 100.0, 0.5)

        # Trophy 2 reads 100% in every read, before and after refreshes,
        # while Chapter 1 is under 1%: a stuck display value, not a rate.
        if last_rate is not None and last_rate >= 99.9:
            rows.append({"id": trophy["id"], "q": artifact_quantiles()})
            continue

        if n_distinct >= 3 and n_target is not None:
            # --- fitted regime: power-law earners, divided by forecast N.
            xs, ys = [], []
            for h, n, rates in obs:
                r = rates[i]
                if r is None:
                    continue
                xs.append(math.log1p(h))
                ys.append(math.log(max(r * n / 100.0, 0.5)))
            if len(xs) >= 3:
                a_e, b_e, resid_e = ols(xs, ys)
                w_e = min(1.0, len(xs) / 10.0)
                b_e = w_e * min(max(b_e, -0.5), 3.0) + (1.0 - w_e) * slope_prior
                log_e_target = a_e + b_e * x_target
                # Cap earners at the category ceiling share of forecast N.
                log_e_cap = math.log(max(ceiling * n_target / 100.0, 0.5))
                log_e_target = min(log_e_target, log_e_cap)
                e_target = math.exp(log_e_target)
                median_en = 100.0 * e_target / n_target
                # Ensemble with a saturating trend fit: logit(rate) linear
                # in ln(1+hours) over the last 24h of distinct reads (the
                # house curve baseline's shape). The power-law E/N forecast
                # overshoots when adoption saturates faster than the
                # category priors assume; the log-time trend is
                # conservative in exactly that case. Averaging the two
                # medians in logit space hedges both failure modes.
                median_tr = trend_median(obs, i, x_target, last_rate)
                if median_tr is not None:
                    median = inv_logit(
                        0.55 * logit(median_en) + 0.45 * logit(median_tr))
                else:
                    median = median_en
                sigma = math.sqrt(
                    resid_e ** 2
                    + 0.35 ** 2 * horizon_days
                    + 0.25 ** 2
                    + spread_extra ** 2
                )
            else:
                median, sigma = prior_forecast(
                    earners_last, last_n, n_target, slope_prior, ceiling,
                    x_last, x_target, horizon_days, spread_extra)
        else:
            # --- refresh regime: history is one stale page. The next
            # refresh multiplies N several-fold (638 -> 3,868 at the first
            # refresh, about 6x) while earners grow only modestly, so the
            # median rate falls to roughly last / 4, with a wide spread.
            n_mult = 5.0
            e_mult = math.exp(slope_prior * 0.35 * math.log1p(horizon_days * 24.0 + 1.0) / 2.0)
            e_mult = min(max(e_mult, 1.0), 2.5)
            median = last_rate * e_mult / n_mult if last_rate > 0 else ceiling * 0.02
            median = min(median, ceiling)
            sigma = math.sqrt(0.9 ** 2 + 0.35 ** 2 * horizon_days + spread_extra ** 2)

        sigma = min(max(sigma, 0.15), 2.2)
        med_logit = logit(median)
        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            qs.append(round4(clamp_pct(q)))
        # Enforce non-decreasing (rounding can only tie, never invert, but
        # keep the guarantee explicit).
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def trend_median(obs, i, x_target, last_rate):
    """Saturating trend median: OLS of logit(rate) on ln(1+hours) over the
    last 24h of distinct reads, read at the target. None when unfittable."""
    last_h = obs[-1][0]
    pts = [(math.log1p(h), logit(rates[i])) for h, _, rates in obs
           if h >= last_h - 24.0 and rates[i] is not None]
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    a, b, _ = ols(xs, ys)
    return clamp_pct(inv_logit(a + b * x_target))


def prior_forecast(earners_last, last_n, n_target, slope_prior, ceiling,
                   x_last, x_target, horizon_days, spread_extra):
    """Fallback median/sigma when a trophy's own series cannot be fit."""
    e_target = earners_last * math.exp(slope_prior * max(0.0, x_target - x_last))
    e_target = min(e_target, max(ceiling * n_target / 100.0, 0.5))
    median = 100.0 * e_target / n_target
    sigma = math.sqrt(0.35 ** 2 * horizon_days + 0.30 ** 2 + spread_extra ** 2)
    return median, sigma


def artifact_quantiles():
    """Trophy 2's stuck 100%: mixture quantiles.

    75% mass near 100% (it survived every refresh so far), 25% mass on the
    story-trophy trajectory (median ~40%) in case the display corrects.
    Quantiles of that mixture, rounded to 4 decimals:
      q05 ~ 22% (correction regime, low tail)
      q25 ~ 58% (top of the correction regime)
      q50 ~ 99.9% (artifact regime)
      q75 ~ 99.95%, q95 ~ 99.99%
    """
    return [22.0, 58.0, 99.9, 99.95, 99.99]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
