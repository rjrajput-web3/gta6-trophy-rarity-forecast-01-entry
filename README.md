# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rjrajput-web3/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players Exophase tracks who will have earned it a week after launch (2026-11-26T00:00:00Z), as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in` case in the input directory and writes `<id>.out` with `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`. Two runs on the same case write identical bytes.

## How it forecasts

A trophy's rate is earners over players-tracked, and the two move for different reasons, so they are forecast separately and divided:

- **Players tracked (N)** grows explosively at launch, then flattens. Exophase refreshes its page on its own schedule, not continuously: it showed 638 tracked players with identical rates from launch night through 04:00Z, then refreshed at 04:30Z to 3,868 players with every rate but the 100% one lower. Reads between refreshes are stale repeats, so consecutive identical reads are collapsed to one effective observation before any fit.
- **Earners (E)** grow as players progress: story trophies fast, grind and collection trophies slow, the platinum and 100% gold slowest. Log-earners are fit against log launch-age, with slopes shrunk toward category priors calibrated on public long-run GTA V trophy data and the launch-day Exophase reads, then capped at the category's week-one ceiling share of forecast N.
- **Ensemble median:** the earners/players forecast is averaged in logit space (55/45) with a saturating trend fit (logit rate linear in log launch-age over the last 24h). The power-law forecast overshoots when adoption saturates faster than the priors assume; the trend fit is conservative in exactly that case.
- **The 100% trophy:** "Just Like Paradise" reads 100.00% in every read, before and after refreshes, while Chapter 1 sits under 1%. No real trophy is held by 100% of tracked players while the first chapter is under 1%, so it is treated as a stuck display value: a mixture, most mass near 100% (it has survived every refresh) and the rest on the story-trophy trajectory in case it corrects.
- **Spread:** per-trophy logit sigma from fit residuals plus a diffusion term growing with the horizon plus a regime term (wide when the history is one stale page, tighter when it is rich).

## Benchmarks (see `benchmark/`)

- **Practice** (answers reconstructed from the house's disclosed 04:30Z refresh: 3,868 players, launch-night earner counts grown 1x–3x): loss **0.24–0.34** vs the house persistence baseline's **0.61–1.69** (the house states ~1.51), typical miss 0.02–0.19 points vs 0.44–0.73, 90% coverage 0.98 vs 0.00.
- **Synthetic final** (structural launch-week simulator, fake f00 cut at the deadline, scored at Nov 26; slow/central/fast adoption): loss **0.25–0.27** vs persistence **0.29–0.30** and the curve baseline **0.26–0.29**.
- **Perturbed final** (simulator ceilings ±50% and adoption half-times ×0.5–×2, so the baked-in priors are wrong): loss **0.24–0.31**, beating persistence in 9/9 runs and the curve baseline in 8/9.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9hqpawrx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
