# NHL Prediction Model — Documentation

This document explains how the model works from the ground up, what each piece does,
and how they all connect. No math background assumed.

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Data Sources](#2-data-sources)
3. [The Dixon-Coles Poisson Model](#3-the-dixon-coles-poisson-model)
4. [Time Weighting (xi)](#4-time-weighting-xi)
5. [Season Blending](#5-season-blending)
6. [Current Season Metrics](#6-current-season-metrics)
7. [Goalie Adjustments](#7-goalie-adjustments)
8. [Back-to-Back Penalty](#8-back-to-back-penalty)
9. [Score Matrix & Probabilities](#9-score-matrix--probabilities)
10. [Puck Line & Totals](#10-puck-line--totals)
11. [Edge Finding (Vig Removal)](#11-edge-finding-vig-removal)
12. [Rolling Backtest](#12-rolling-backtest)
13. [Daily Workflow](#13-daily-workflow)
14. [Key Parameters at a Glance](#14-key-parameters-at-a-glance)

---

## 1. Big Picture

The model answers one question: **given two teams, what is the probability that each team wins?**

From that probability you can compare to the sportsbook's implied probability and find "edges" —
spots where you think a team is more likely to win than the book is giving credit for.

The model is built in layers:

```
Prior season W/L data
        +
Current season W/L data        →  Blend  →  Team attack/defense ratings
        +                                            |
Current season xG/shot metrics                       |
        +                                            ↓
Goalie GSAx ratings                         Expected goals per team
        +                                   (lambda_home, lambda_away)
Back-to-back flag                                    |
                                                     ↓
                                          Full score matrix (0-0, 1-0, ...)
                                                     |
                                                     ↓
                              Win%, OT%, Puck Line%, Over/Under%
```

---

## 2. Data Sources

| Source | What it provides | Where used |
|--------|-----------------|------------|
| `[dbo].[NHL_Gamelog]` (MoneyPuck) | Game-by-game team stats: goals, xG, Corsi, shot locations | Training the Dixon-Coles model and metrics |
| `[dbo].[NHL_Goalie_Stats]` (MoneyPuck) | Goalie GSAx (Goals Saved Above Expected) | Goalie rating adjustment |
| `nhl-202526-asplayed.csv` | Schedule, scores, and market odds for each game | B2B detection, backtest, edge finder |

The prior season data is the 2024-25 season. Current season is 2025-26.

---

## 3. The Dixon-Coles Poisson Model

### What is a Poisson model?

In hockey, goals are somewhat random. The **Poisson distribution** models how many times a rare event
(a goal) happens over a fixed time period. Each team gets a single number — their **expected goals (lambda)** —
and the model says: "this team will score around lambda goals per game on average."

### What is Dixon-Coles?

Dixon-Coles (1997) is a classic improvement on the basic Poisson model. It adds two things:

1. **Team-specific ratings** — every team has an `attack` rating (how much they score) and a
   `defense` rating (how much they allow). These are in log-space (explained below).

2. **Low-score correction (rho)** — basic Poisson over-predicts 0-0 draws and under-predicts 1-0,
   0-1, and 1-1 results. Dixon-Coles adds a correction factor (`rho`) to fix this for those four
   specific scorelines.

### How expected goals are calculated

```
lambda_home = exp(log_mu + home_adv + attack_home - defense_away)
lambda_away = exp(log_mu            + attack_away - defense_home)
```

Breaking down each term:

- **`log_mu`** — the league-average log goals per game. Acts as a baseline.
- **`home_adv`** — home ice advantage, added only for the home team.
- **`attack_home`** — how much above/below average the home team scores (positive = good offence).
- **`defense_away`** — how much the away team leaks goals (positive = bad defence).
- **`defense_home`** — subtracted from the away team's expected goals.

Everything is in **log-space** so you can add terms instead of multiply. When you take `exp()` at
the end, you get back to a real number of expected goals.

### Why log-space?

Because you want attack and defense to combine multiplicatively (a great offence against a terrible
defence should be a big effect), but addition is easier for the optimizer to work with. Log-space
lets you add things that will multiply in real-space.

### How the model is "fit" (MLE)

The model has ~65 parameters (attack + defense for 32 teams + home advantage + log_mu + rho).
We use **Maximum Likelihood Estimation (MLE)** — find the parameter values that make the observed
historical scores as probable as possible.

Technically, for each historical game, we compute:

```
P(game happened) = Poisson(home_score | lambda_h)
                 × Poisson(away_score | lambda_a)
                 × dc_correction(rho)
```

We sum the log of this across all games (weighted by recency) and find the parameters that
maximize it. This is done by `scipy.optimize.minimize` using the L-BFGS-B algorithm.

---

## 4. Time Weighting (xi)

A game from October is less relevant than a game from last week. We apply an **exponential decay**:

```
weight = exp(-xi × days_ago)
```

With `xi = 0.005`:
- A game from **yesterday** gets weight ≈ **1.0** (full weight)
- A game from **4 months ago** (~120 days) gets weight ≈ **0.55** (55% weight)
- A game from **last season** (365+ days ago) gets weight close to **0**

Each game's contribution to the log-likelihood is multiplied by this weight before summing.
Older games still count, they just count less.

---

## 5. Season Blending

At the start of a new season there isn't enough data to trust the current season ratings alone.
We blend three sources:

```
final_rating = (prior_weight    × prior_season_DC)
             + (curr_wl_weight  × current_season_DC)
             + (curr_met_weight × current_season_metrics)
```

The weights shift automatically based on how many current-season games have been played:

| Games played | Prior weight | W/L DC weight | Metrics weight |
|:---:|:---:|:---:|:---:|
| < 10 | 80% | 4% | 16% |
| 10–19 | 50% | 25% | 25% |
| 20–29 | 20% | 60% | 20% |
| 30–39 | 0% | 75% | 25% |
| 40+ | 0% | 40% | 60% |

By mid-season (40+ games), the prior season is fully dropped and metrics get more weight than
raw W/L results because xG-based metrics are a better predictor of future performance.

---

## 6. Current Season Metrics

The metrics component (`CurrentSeasonMetrics`) uses **advanced stats** instead of just wins/losses:

| Stat | Weight | Why |
|------|--------|-----|
| xGF per game (expected goals for) | 50% | Best predictor of future goals |
| High-danger xGF per game | 30% | Quality chances, more signal |
| Shots on goal for per game | 20% | Volume indicator |

Same three stats for the defensive side (xGA, HD xGA, SOG against).

Each stat is **z-scored** (converted to standard deviations from the league average) so they're
on the same scale before being blended. The final z-score is then scaled by 0.10 to keep the
adjustment in a reasonable log-space range.

This means a team with great xG numbers gets a slight boost to their attack rating even if their
win-loss record hasn't caught up yet.

---

## 7. Goalie Adjustments

Goalies are rated using **GSAx (Goals Saved Above Expected)** — how many goals they saved compared
to what an average goalie would have saved, given the shot quality they faced.

```
GSAx = xGoals_faced - Goals_allowed
```

A positive GSAx means the goalie is better than average (saved more than expected).
A negative GSAx means the goalie is below average.

### How it's applied

The goalie adjustment works by reducing the **opponent's** expected goals:

```
lambda_home -= away_goalie_adjustment   # home team faces the away goalie
lambda_away -= home_goalie_adjustment   # away team faces the home goalie
```

The log-space adjustment is:

```
log_adj = clip(GSAx_per_game, -0.40, +0.40) × 0.33
```

So a goalie saving 0.3 extra goals per game gets a `log_adj ≈ +0.10`, reducing the opponent's
expected goals by about 10%.

The `0.33` scaling factor is intentional — we don't want to over-trust individual goalie
performance since it's noisy. Goalies must have played at least 5 games to be rated.

---

## 8. Back-to-Back Penalty

If a team played last night and is playing again today, they are on a **back-to-back (B2B)**.
Fatigue reduces performance. The model applies a penalty of `B2B_PENALTY = -0.04` in log-space:

```
lambda_home *= exp(-0.04)  ≈  lambda_home × 0.961
```

In plain terms, a B2B team's expected goals are reduced by about **4%**. Small but meaningful —
roughly 0.13 fewer expected goals for a team averaging 3.0 per game.

---

## 9. Score Matrix & Probabilities

Once we have `lambda_home` and `lambda_away`, we build a **score matrix** — a grid of all possible
final scores from 0-0 to 15-15, with the probability of each scoreline.

```
P(home scores gh, away scores ga) = Poisson(gh | lam_h) × Poisson(ga | lam_a) × dc_rho
```

The matrix is 16×16 = 256 cells. Each cell is a probability. They sum to 1.

From the matrix we read off:

| Outcome | How it's calculated |
|---------|-------------------|
| **Home win (regulation)** | Sum of all cells where home > away |
| **Away win (regulation)** | Sum of all cells where away > home |
| **Draw (→ OT)** | Sum of the diagonal (0-0, 1-1, 2-2, ...) |
| **Home win% (final)** | Home reg + 50% of OT |
| **Away win% (final)** | Away reg + 50% of OT |

The 50% OT split is a simplification — in reality home teams win OT slightly more often, but
the difference is small.

---

## 10. Puck Line & Totals

### Puck Line (±1.5)

Directly from the score matrix:

```
Home -1.5 covers  =  P(home_score - away_score >= 2)
Away +1.5 covers  =  everything else
```

Just sum the cells in the top-right triangle (home wins by 2+).

### Totals (O/U)

Loop through every cell in the matrix, sort by whether `gh + ga` is over, under, or exactly equal
to the O/U line. Push probability (exact tie with the line) is redistributed proportionally to
over and under:

```
over_fair  = over_prob  / (over_prob + under_prob)
under_fair = under_prob / (over_prob + under_prob)
```

This gives the "true" over/under probability after removing the push.

---

## 11. Edge Finding (Vig Removal)

Sportsbooks charge a **vig** (juice) — they build a profit margin into the odds. A -110/-110 line
implies each side has a 52.4% probability, but they can't both be 52.4% (they must sum to 100%).

To find the **fair implied probability** we remove the vig:

```
implied_home = american_to_prob(home_ml)
implied_away = american_to_prob(away_ml)
total        = implied_home + implied_away

fair_home = implied_home / total
fair_away = implied_away / total
```

**Edge** is simply:

```
edge = model_probability - fair_implied_probability
```

A positive edge means the model thinks the team is more likely to win than the fair odds suggest.

Example:
- Model says home team wins 58%
- Book is -150 (fair implied ≈ 56%)
- Edge = 58% - 56% = **+2%**

---

## 12. Rolling Backtest

The backtest simulates using the model "in real-time" through the season. For each week:

1. **Train** on all data available *before* that week (as if you're standing on that date)
2. **Predict** that week's games
3. **Record** the prediction vs the actual result

At the end, compare model probabilities to Vegas implied probabilities using:

| Metric | What it measures |
|--------|-----------------|
| **Brier Score** | Mean squared error of probabilities. Lower = better. Random = 0.25. |
| **Log-Loss** | Penalises confident wrong predictions more. Lower = better. |
| **Accuracy** | % of games where the predicted favourite actually won. |

The calibration curve shows whether the model's probabilities are realistic — if it says 60%,
does the team actually win 60% of the time?

> **Note:** The backtest uses full-season goalie ratings, which is a minor "lookahead bias."
> This means backtest metrics are slightly optimistic. Acceptable for an initial evaluation.

---

## 13. Daily Workflow

### Morning of game day:

1. **Update the schedule CSV** with today's games (if not already there)
2. **Launch the edge finder** — `streamlit run nhl_app.py`
3. Enter the market odds and goalies for each game
4. Click **Run Predictions**
5. Review the three tabs: Moneyline, Puck Line, Totals

### Weekly (or when you want updated ratings):

1. **Re-scrape the gamelog** — run the MoneyPuck scraper notebook to pull in new games
2. **Re-train the model** — run the main notebook cells 1–10
3. The model is automatically saved to `nhl_skellam_model.json`
4. The Streamlit app will load the new model next time (or click **Reload model** in the sidebar)

---

## 14. Key Parameters at a Glance

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `xi` | 0.005 | Time decay rate. Higher = more weight on recent games. |
| `B2B_PENALTY` | -0.04 | Log-space reduction for back-to-back teams (~4% fewer goals). |
| Goalie scale | 0.33 | How much GSAx translates into expected goal reduction. |
| Goalie clip | ±0.40 | Max/min GSAx per game allowed (prevents outliers dominating). |
| Min games (goalie) | 5 | Goalie must have 5+ games to be rated. |
| `maxiter` | 300 | Max optimizer iterations per MLE fit. Higher = slower but more precise. |
| `ftol` | 1e-7 | Optimizer convergence tolerance. |
| Score matrix size | 15 max goals | Grid is 16×16 = 256 scoreline cells. |
| Metrics scale | 0.10 | How much advanced stats z-scores shift attack/defense in log-space. |

---

*Model built in Python using scipy, pandas, numpy. Data from MoneyPuck.*
