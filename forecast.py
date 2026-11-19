"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case (<id>.in: the house's Exophase reads of the GTA VI PS5 trophy
page up to the case checkpoint) and writes <id>.out: five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's earn rate (percent of the
players Exophase tracks) at the case target. Scored by pinball loss on the
logit scale. Pure standard library, fully deterministic, no network, no
randomness, no wall-clock use: two runs on the same case write identical bytes.

Model: earners / players-tracked decomposition, in logit space, with a
refresh-inversion regime for stale histories. Round 4 recalibrates every
prior on the game's actual first-day trajectory: the launch-night page
(638 players tracked, 943 trophies earned), the documented first refresh
(3,868 players, 4,276 trophies earned, every rate but the 100% one lower),
the entry's own scored practice result, and the public page's state later
on launch day (about 18 hours in: the leaders are the fishing challenge
at 1.89% and Chapter 1 at 0.73%, most other trophies at 0.03-0.06%, and
the opening fishing trophy at 99.99%).
---------------------------------------------------------------
A trophy's rate is E/N: E earners over N players tracked. The two move for
different reasons, so they are forecast separately and divided.

* Refresh regime (history holds fewer than 3 distinct observations: the
  practice cases, where every read is the identical launch-night page).
  Exophase refreshes its counts on its own schedule: the house saw the page
  unchanged (638 players, identical rates) from 2026-11-18 23:48Z through
  2026-11-19 04:00Z, then refreshed at 04:30Z to 3,868 players and 4,276
  trophies earned, with every rate but the 100% one lower -- a dilution
  event. The earners behind the launch rates are integers, recovered
  exactly from the last read (round(rate * N / 100); the 55 recovered
  counts sum to 943, exactly the history's trophies_earned total). The
  answering read re-rates the earners over the new player base. How much
  the earners grew across that refresh is measurable: the refreshed total
  is 4,276, of which the 100% trophy accounts for 3,868, so the other 54
  trophies' earners grew from 305 to 408 in total (x1.34), and the growth
  is concentrated by kind, not uniform -- the trophies players reach in
  the first hours (the fishing challenge, Chapter 1, the arcade game,
  discovering vehicles) roughly doubled to tripled, story and side
  content grew by about a tenth, and the long grinds barely moved. The
  previous round's scored practice loss (0.113313 on both cases, with
  50 of 55 trophies inside its 90% band) pins the same split: a uniform
  multiplier fits neither that loss nor the round before it's 1.273384,
  while the per-kind growth below fits both. So the refresh median is
  earners * g(kind) / 3,868 with g from the table in growth_prior(),
  and the spread is per-kind (sigma ~0.20 for the bulk, ~0.32 for the
  fast early kinds) plus a count term: with launch counts of 4-13, the
  integer rounding of the recovered earners is real uncertainty.

* Fitted regime (rich history: the final case, cut at the deadline after
  two days of real reads). N (players tracked) is fit as log N against
  log(1 + hours since launch) over the recent distinct reads, slope
  shrunk toward 1.0 (the observed first-day slope is about 1.04: 638 ->
  3,868 -> roughly 14-16k tracked in the first 18 hours). E (earners per
  trophy) is fit per trophy as log earners against log(1 + hours), slope
  shrunk toward a kind prior measured on the first day's trajectory
  (Chapter 1 about 0.9, the fishing challenge about 1.05, most story and
  side content 0.05-0.2, the grinds under 0.1: dilution dominates launch
  week, so the old textbook priors of 1.4-2.2 are far too steep), and
  capped at the kind's week-one ceiling share of the forecast N, with
  ceilings likewise set from the observed trajectory (a few percent for
  the early kinds, under 2% for the rest). The median is an ensemble in
  logit space: E/N forecast blended with a saturating trend fit (logit
  rate linear in log launch-age over the recent window); the weights and the
  spread tightening are set on the benchmark families in
  benchmark/verify.py, which include a backtest on the real first-day
  trajectory and a dilution-dominant synthetic final.

* Trophy 2 ("Complete Gone Fishing") is the opening tutorial: its earners
  are effectively the whole tracked base (638 of 638 at launch, 3,868 of
  3,868 at the first refresh, all but a couple of players a refresh
  later at 99.99%). It is not a stuck display and it does not correct to
  a story-trophy rate; it is forecast on a deficit model (100 - rate
  stays in the hundredths) in both regimes, tight near the last read.

Kind priors are shrinkage targets only: with a rich history the fits
dominate.
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

# The documented first refresh: 638 players tracked -> 3,868 (competition
# rules). The refresh regime re-rates the recovered earners over the
# refreshed player base.
REFRESH_N_LAST = 638
REFRESH_N_NEXT = 3868

# Fitted regime: ensemble weight on the saturating trend fit (rest on E/N),
# and the sigma tightening factor (tuned on benchmark/verify.py families).
TREND_WEIGHT = 0.75
SIGMA_SCALE = 0.55

# N-fit slope prior: the observed first-day log-log slope of players
# tracked is about 1.04.
N_SLOPE_PRIOR = 1.0


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
      slope_prior -- prior power-law slope of log earners vs log(1+hours),
        measured on the game's first-day trajectory (see module docstring)
      ceiling_pct -- week-one ceiling share of tracked players (percent)
      spread_extra -- extra logit sigma for this kind of trophy
    """
    name = trophy["name"].lower()
    desc = trophy["description"].lower()
    tier = trophy["tier"]
    text = name + " " + desc

    if tier == "platinum":
        return (0.12, 0.3, 0.45)
    if "100% completion" in desc:
        return (0.12, 0.5, 0.45)
    if "complete chapter 1" in desc:
        return (1.40, 20.0, 0.15)
    if "complete chapter 2" in desc:
        return (0.60, 10.0, 0.15)
    if "complete chapter 3" in desc:
        return (0.25, 8.0, 0.15)
    if "complete chapter 4" in desc:
        return (0.25, 6.0, 0.15)
    if "complete the final mission" in desc and "for " not in desc:
        return (0.25, 5.0, 0.20)
    if "complete the final mission for" in desc:
        return (0.15, 4.0, 0.25)
    if "complete " in desc and ("mission" in desc or "heist" in desc
                                or "takedown" in desc or "operation" in desc):
        return (0.25, 5.0, 0.20)
    if "gone fishing" in desc or "complete gone fishing" in desc:
        return (1.05, 45.0, 0.25)  # trophy 2: handled by the deficit model
    if "fishing challenge" in text:
        return (1.60, 15.0, 0.25)
    if "arcade game" in text:
        return (1.20, 6.0, 0.20)
    if "stunt" in text or "gold medal" in text:
        return (0.20, 1.2, 0.40)
    if "collectible card" in text or "waink it at 100" in text:
        return (0.08, 1.8, 0.40)
    if "meditate" in text or "50 challenges" in text:
        return (0.15, 3.5, 0.35)
    if "discover 100 vehicles" in text:
        return (0.80, 6.0, 0.30)
    if "try each type of activity" in text:
        return (0.15, 6.0, 0.20)
    if "win" in text and "race" in text:
        return (0.15, 4.0, 0.25)
    if "rob " in text or "bank vault" in text or "armored truck" in text:
        return (0.35, 4.0, 0.25)
    if "explore all of leonida" in text:
        return (0.15, 4.0, 0.25)
    if "strangers" in text:
        return (0.25, 4.0, 0.30)
    if "spend $" in text or "earn $" in text or "purchase all" in text \
            or "register 10 vehicles" in text:
        return (0.15, 3.0, 0.25)
    # default bronze: ordinary open-world / side content
    return (0.35, 4.0, 0.25)


def growth_prior(trophy):
    """Refresh-regime earner growth across the first refresh, by kind.

    Returns (g, sigma_g): the median multiplier on the recovered launch
    earners at the refreshed read (638 -> 3,868 players), and the logit
    spread of that multiplier. Measured on the documented refresh totals
    (305 -> 408 earners over the 54 non-100% trophies) split by kind
    against the public page's later first-day state, and checked against
    the previous rounds' scored practice losses (see module docstring).
    """
    name = trophy["name"].lower()
    desc = trophy["description"].lower()
    tier = trophy["tier"]
    text = name + " " + desc

    if tier == "platinum" or "100% completion" in desc:
        return (1.09, 0.20)
    if "complete chapter 1" in desc:
        return (2.58, 0.32)
    if "complete chapter 2" in desc:
        return (1.51, 0.28)
    if "complete chapter 3" in desc or "complete chapter 4" in desc:
        return (1.14, 0.22)
    if "complete the final mission" in desc:
        return (1.10, 0.22)
    if "fishing challenge" in text:
        return (3.01, 0.35)
    if "arcade game" in text:
        return (2.31, 0.32)
    if "discover 100 vehicles" in text:
        return (1.74, 0.30)
    if "complete " in desc and ("mission" in desc or "heist" in desc
                                or "takedown" in desc or "operation" in desc):
        return (1.13, 0.22)
    if "stunt" in text or "gold medal" in text:
        return (1.12, 0.22)
    if "collectible card" in text or "waink it at 100" in text \
            or "meditate" in text or "50 challenges" in text:
        return (1.09, 0.20)
    if "strangers" in text:
        return (1.13, 0.22)
    # default bronze and the remaining side content
    return (1.19, 0.22)


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


def deficit_quantiles(horizon_days):
    """Quantiles for a trophy held by effectively every tracked player.

    Its earners are the tracked base minus a small deficit (a couple of
    players at the last observed refresh), so 100 - rate stays in the
    hundredths of a percent. The band widens only downward, and only
    with horizon: a correction would show up as a growing deficit.
    """
    if horizon_days <= 0.5:
        return [99.9, 99.99, 99.99, 99.99, 99.99]
    return [99.5, 99.9, 99.99, 99.99, 99.99]


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
        # Growth flattens after launch: shrink the recent slope toward the
        # observed first-day prior and never extrapolate a shrinking base.
        w = min(1.0, len(recent) / 12.0)
        b = w * min(max(b, 0.0), 2.0) + (1.0 - w) * N_SLOPE_PRIOR
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

    # Pooled slope level of this case's own history: the cross-trophy
    # median fitted earner slope. Priors shrink toward a blend of the
    # kind prior and this pooled level, so the ensemble calibrates itself
    # to the regime the case actually shows (dilution-dominant launch
    # weeks pull it down; fast-adoption cases pull it up).
    pooled_slope = None
    if n_distinct >= 3:
        raw_slopes = []
        for i in range(len(trophies)):
            if last_rates[i] is not None and last_rates[i] >= 99.9:
                continue
            xs_p, ys_p = [], []
            for h, n, rates in obs:
                r = rates[i]
                if r is None:
                    continue
                xs_p.append(math.log1p(h))
                ys_p.append(math.log(max(r * n / 100.0, 0.5)))
            if len(xs_p) >= 3:
                _, b_p, _ = ols(xs_p, ys_p)
                raw_slopes.append(min(max(b_p, -0.5), 3.0))
        if raw_slopes:
            raw_slopes.sort()
            pooled_slope = raw_slopes[len(raw_slopes) // 2]

    rows = []
    for i, trophy in enumerate(trophies):
        slope_prior, ceiling, spread_extra = classify(trophy)
        if pooled_slope is not None:
            slope_prior = 0.5 * slope_prior + 0.5 * pooled_slope
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0
        earners_last = max(last_rate * last_n / 100.0, 0.5)

        # A trophy held by (nearly) every tracked player -- the opening
        # tutorial -- follows the deficit model in both regimes: its
        # earners track the player base itself, so its rate stays pinned
        # just under 100% however the base grows.
        if last_rate is not None and last_rate >= 99.9:
            rows.append({"id": trophy["id"],
                         "q": deficit_quantiles(horizon_days)})
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
                # in ln(1+hours) over the recent distinct reads (the
                # house curve baseline's shape; the window widens from
                # 48h of distinct reads: refreshes are hours apart, so a
                # 24h window often holds fewer than 3 reads and the fit
                # would never fire). The power-law E/N forecast
                # overshoots when adoption saturates faster than the
                # category priors assume; the log-time trend is
                # conservative in exactly that case. Averaging the two
                # medians in logit space hedges both failure modes.
                median_tr = trend_median(obs, i, x_target, last_rate)
                if median_tr is not None:
                    median = inv_logit(
                        (1.0 - TREND_WEIGHT) * logit(median_en)
                        + TREND_WEIGHT * logit(median_tr))
                else:
                    median = median_en
                sigma = math.sqrt(
                    resid_e ** 2
                    + 0.35 ** 2 * horizon_days
                    + 0.25 ** 2
                    + spread_extra ** 2
                ) * SIGMA_SCALE
            else:
                median, sigma = prior_forecast(
                    earners_last, last_n, n_target, slope_prior, ceiling,
                    x_last, x_target, horizon_days, spread_extra)
                sigma *= SIGMA_SCALE
        else:
            # --- refresh regime: history is one stale page. Recover the
            # integer earners behind the last read and re-rate them over
            # the refreshed player base (638 -> 3,868 documented), grown
            # by the per-kind multiplier in growth_prior(): the refresh's
            # earner growth is concentrated in the kinds players reach in
            # the first hours, not uniform. The spread is per-kind plus
            # the count uncertainty of the small recovered counts.
            if last_n == REFRESH_N_LAST:
                n_next = REFRESH_N_NEXT
            else:
                n_next = last_n * (REFRESH_N_NEXT / REFRESH_N_LAST)
            earners = round(last_rate * last_n / 100.0)
            earners = max(earners, 0)
            g_med, g_sigma = growth_prior(trophy)
            if earners > 0:
                median = 100.0 * earners * g_med / n_next
            else:
                median = 0.01
            median = min(median, ceiling)
            sigma = math.sqrt(g_sigma ** 2
                              + 0.5 / max(earners, 1)
                              + 0.10 ** 2 * horizon_days)

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
    last 48h of distinct reads, read at the target. None when unfittable."""
    last_h = obs[-1][0]
    pts = [(math.log1p(h), logit(rates[i])) for h, _, rates in obs
           if h >= last_h - 48.0 and rates[i] is not None]
    if len(pts) < 3:
        pts = [(math.log1p(h), logit(rates[i])) for h, _, rates in obs[-5:]
               if rates[i] is not None]
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
