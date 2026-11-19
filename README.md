# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
competition `gta6-trophy-rarity-forecast-01`: forecast, for every Grand Theft
Auto VI PlayStation 5 trophy, its earn rate on Exophase at 2026-11-26T00:00:00Z
as five quantiles (5%, 25%, 50%, 75%, 95%), scored by mean pinball loss on the
logit scale.

## How it forecasts

`main.py` is the packed entrypoint (built from the readable `forecast.py` with
the kit's `makefun compress`). For each trophy it works on the logit scale:

1. **Recent trend.** A least-squares line of logit(rate) against
   log(1 + hours since the first read), fitted on the most recent 24 hours of
   reads and extrapolated to the target, with the slope damped toward flat.
2. **Own history.** The damped trend is blended with the trophy's last
   observed rate, the strongest signal when the history is flat or short.
3. **Tier shrinkage.** The blend is pulled toward the median rate of the
   trophy's own tier (platinum, gold, silver or bronze); the pull grows with
   the horizon, so a 4.5-day forecast leans on what trophies of the same kind
   usually do.
4. **Honest spread.** The quantile width grows with the horizon, the fit
   residual, and an extra term for trophies pinned at 100%, whose rate can
   only fall once new players arrive.

Pure Python standard library, fully deterministic, no network, no randomness.

## Local check

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 benchmark/verify.py
```

`benchmark/verify.py` runs synthetic out-of-sample holdouts (three simulated
launches, cut at hour 48 and scored against the true rates at hour 168) and
validates the entry on the kit's practice cases p00/p01, including
byte-identical output across two runs. On the synthetic holdouts the forecast
beats both house baselines on every seed (mean pinball loss about 0.21 vs
about 0.33 for persistence and about 0.31 for the fitted curve).

## Layout

- `forecast.py` — readable source of the forecaster
- `main.py` — packed entrypoint, produced by `makefun compress`
- `benchmark/verify.py` — synthetic holdout benchmark + practice validation
- `LICENSE` — MIT
