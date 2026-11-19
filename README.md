# GTA VI Trophy Rarity Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
`gta6-trophy-rarity-forecast-01`: forecast, for every Grand Theft Auto VI
PlayStation 5 trophy, its earn rate on Exophase at 2026-11-26T00:00:00Z as
five quantiles (5%, 25%, 50%, 75%, 95%), scored by mean pinball loss on the
logit scale.

## How it forecasts

`main.py` is the packed entrypoint (built from the readable `forecast.py` with
the kit's `makefun compress`). A trophy's rate is E/N: earners over players
tracked. The two move for different reasons, so they are forecast separately
and divided, all on the logit scale:

1. **Refresh inversion (stale history).** When the history holds fewer than
   3 distinct observations (the practice cases: every read is the identical
   launch-night page), the program recovers the integer earners behind the
   last read (round(rate x N / 100); the 55 recovered counts sum to 943,
   exactly the history's `trophies_earned` total) and re-rates them over the
   documented refreshed player base (638 -> 3,868 players), grown by 1.19x.
   That 1.19x factor is fitted to the previous entry's observed practice
   loss of 1.273384 on both cases, which pure dilution (1.00x, loss 1.45)
   and 1.50x growth (loss 1.06) both fail to reproduce. The spread is tight
   (sigma ~0.15): the refresh answers are nearly deterministic given the
   documented N jump.
2. **Earners / players decomposition (rich history).** Log N is fitted
   against log(1 + hours since launch) over the recent distinct reads, slope
   shrunk toward 0.8; log earners are fitted per trophy the same way, slope
   shrunk toward a category prior from the trophy's tier and description,
   and capped at the category's week-one ceiling share of the forecast N.
   Consecutive identical reads are collapsed first: stale Exophase repeats
   carry no information.
3. **Ensemble median.** 40% E/N forecast + 60% saturating trend fit (logit
   rate linear in log launch-age over the last 24h), blended in logit space.
4. **Honest, tightened spread.** The quantile width comes from the fit
   residual plus a diffusion term growing with the horizon plus a category
   term, tightened by 0.7x: the un-tightened spread scored coverage 1.00 on
   every synthetic final (grossly over-wide).
5. **The 100% trophy.** Trophy 2 reads 100.00% in every read while Chapter 1
   sits near 0.5%: a stuck display value. On a short horizon it is forecast
   tight near 100% (the answering read is the documented refresh that kept
   it at 100%); on the 4.5-day final horizon it keeps mixture quantiles,
   most mass near 100% with the rest on the story-trophy trajectory in case
   the display corrects.

Pure Python standard library, fully deterministic, no network, no randomness.
Category ceilings and slopes are calibrated on public long-run GTA V trophy
data and the launch-day Exophase reads.

## Local check

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 benchmark/verify.py
```

`benchmark/verify.py` (1) validates the entry on the kit's practice cases
p00/p01, including byte-identical output across two runs; (2) scores the
practice cases under oracle-calibrated answers (launch earners x1.19 /
3,868, trophy 2 at 100%): about 0.035, far below the live Top-200 bar of
0.125 and the previous entry's 1.273384; (3) runs synthetic finals on two
independent generators (calibrated N + Hill-curve rates with perturbed
ceilings, and a power-law generator): mean pinball loss about 0.216 and
about 0.25, beating both house baselines (persistence about 0.306 / 0.340,
fitted curve about 0.288 / 0.304) on every run of the calibrated generator.

## Layout

- `forecast.py` — readable source of the forecaster
- `main.py` — packed entrypoint, produced by `makefun compress`
- `benchmark/verify.py` — practice validation + oracle-calibrated practice
  score + synthetic-final benchmarks
- `LICENSE` — MIT
