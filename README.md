# NFL Offensive Player Data: Analysis and Prediction

**Authors:** Austin Bennett and Jared Weissberg

In this repository, we analyze offensive player data in the NFL in R. Final results are presented up front. We utilize a ... model.

## Table 1. Final 2021 (Effecitvely 2022 Predictions) Holdout Results by Position

| Pos | Best Model | MSE (2021) | RMSE (2021)† |
|:-------:|:-------------:|---------------:|----------------:|
| QB | Ridge         | 6308.36        | 79.45           |
| RB  | Ridge         | 5387.21        | 73.41           |
| WR  | XGB           | 3329.36        | 57.71           |
| TE  | XGB           | 1500.16        | 38.73           |

> **Notes**  
> \* **Mean RMSE (CV)** is the root mean squared error averaged over 11 year-based folds (2011–2021).  
> † **Final Test RMSE** is from training on 2010–2020 and testing on the 2021 data (predicting 2022 PPR).

---

## Table 2. Top-10 Predicted vs. Top-10 Actual (Any Position)

### Top-10 Predicted

| Pred. Rank | Player       | Pos | Actual PPR | Predicted PPR |
|---------------:|:-----------------|:-------:|---------------:|-------------------:|
| 1              | Josh Allen       | QB      | 435            | 393               |
| 2              | Patrick Mahomes  | QB      | 478            | 375               |
| 3              | Tom Brady        | QB      | 294            | 346               |
| 4              | Travis Kelce     | TE      | 393            | 343               |
| 5              | Matthew Stafford | QB      | 108            | 337               |
| 6              | Ja'Marr Chase    | WR      | 297            | 311               |
| 7              | Justin Herbert   | QB      | 297            | 310               |
| 8              | Davante Adams    | WR      | 336            | 308               |
| 9              | Joe Burrow       | QB      | 407            | 304               |
| 10             | Tyreek Hill      | WR      | 358            | 302               |

### Top-10 Actual

| Actual Rank | Player            | Pos | Actual PPR | Predicted PPR |
|----------------:|:----------------------|:-------:|---------------:|-------------------:|
| 1               | Patrick Mahomes       | QB      | 478            | 375               |
| 2               | Jalen Hurts           | QB      | 458            | 286               |
| 3               | Josh Allen            | QB      | 435            | 393               |
| 4               | Christian McCaffrey   | RB      | 416            | 112               |
| 5               | Joe Burrow            | QB      | 407            | 304               |
| 6               | Travis Kelce          | TE      | 393            | 343               |
| 7               | Austin Ekeler         | RB      | 391            | 222               |
| 8               | Justin Jefferson      | WR      | 380            | 262               |
| 9               | Tyreek Hill           | WR      | 358            | 302               |
| 10              | Stefon Diggs          | WR      | 342            | 197               |

**Overlap in Top-10**: 5 players  
**Points by predicted Top-10**: 3401.1  
**Points by actual Top-10**: 4057.0  
**Difference**: 655.9  

---

## Table 3. “Standard Roster” (1 QB, 2 RB, 2 WR, 1 TE) Comparison (Predicted vs. Actual)

| Roster               | Score |
|:------------------------:|----------:|
| Model’s Predicted    | 1881.2    |
| Best Actual          | 2415.7    |
| Difference           | 534.5     |

---

Below is a brief description of each file:

- [**summary_stats.R**](https://github.com/weissbergj/NFL/blob/main/summary_stats.R)  
  Provides initial data visualizations and summary statistics for 2022 weekly data.

- [**summary_stats.log**](https://github.com/weissbergj/NFL/blob/main/summary_stats.log)  
  Contains the console output from running **summary_stats.R**.

- [**attempt1.R**](https://github.com/weissbergj/NFL/blob/main/attempt1.R)  
  An initial script experimenting with linear regressions and k-means clustering. It does not predict but carries over methods from earlier assignments.

- [**prediction.R**](https://github.com/weissbergj/NFL/blob/main/prediction.R)  
  A first attempt at predictive modeling using linear regression, LASSO, Ridge, random forest, and XGBoost. Results are logged in **output.log**.

  - [**output.log**](https://github.com/weissbergj/NFL/blob/main/output.log)  
  Contains the console output from running **prediction.R**.

- [**prediction2.R**](https://github.com/weissbergj/NFL/blob/main/prediction2.R)  
  Extends the predictive approach with additional features and computes feature importance. It also builds and evaluates a team roster. Results are logged in **output2.log**.

- [**output2.log**](https://github.com/weissbergj/NFL/blob/main/output2.log)  
  Contains the console output from running **prediction2.R**.

- [**robust.R**](https://github.com/weissbergj/NFL/blob/main/output.log)  
  The final model script that does individual position equations along with more robust features (e.g., injuries) from other datasets.

- [**robust1.log**](https://github.com/weissbergj/NFL/blob/main/robust1.log)  
  Contains the console output from running **robust.R**.

- [**robust2.R**](https://github.com/weissbergj/NFL/blob/main/obust2.R)  
  An unsuccessful attempt to do a significantly more robust hyperparemeter sweep using caret along with ensembling.

- [**robust2.log**](https://github.com/weissbergj/NFL/blob/main/robust2.log)  
  Contains the console output from running **robust2.R**.

- [**QB.R**](https://github.com/weissbergj/NFL/blob/main/QB.R)  
  An unsuccessful attempt at adding QB-specific features to reduce MSE.

- [**Rplots.pdf**](https://github.com/weissbergj/NFL/blob/main/Rplots.pdf)  
  Displays initial data visualizations, which are not highly informative.

Additional results from previous runs are below.

## Table 4. Results from `output.log` (First Predictive Approach, **RMSE**)

| Model         | Mean RMSE (CV)* | Final Test RMSE† |
|---------------|-----------------|-------------------|
| Linear        | 68.57           | **63.42**        |
| Lasso         | 68.59           | 63.44            |
| **Ridge**     | **68.38**       | 63.77            |
| RandomForest  | 70.61           | 65.16            |
| XGBoost       | 74.66           | 69.38            |

---

## Table 5. Results from `output2.log` (Second Predictive Approach, **RMSE**)

| Model         | Mean RMSE (CV)* | Final Test RMSE† |
|---------------|-----------------|-------------------|
| Linear        | 69.09           | 63.64            |
| Lasso         | 68.55           | **63.63**        |
| Ridge         | **68.41**       | 63.64            |
| Elastic Net   | 68.42           | **63.63**        |
| RandomForest  | 70.63           | 64.89            |
| XGBoost       | 73.29           | 66.87            |

---

## Table 6. RMSE by Position (Second Predictive Approach on Final Test Set (2021 → 2022))

| Position | Linear | Lasso | Ridge | Elastic Net | RandomForest | XGBoost |
|----------|-------:|------:|------:|-----------:|------------:|--------:|
| **QB**   | 80.93  | 81.23 | 80.95 | 81.00       | 84.96        | 93.72   |
| **RB**   | 73.09  | 72.83 | 73.01 | 72.97       | 73.31        | 73.43   |
| **TE**   | 40.59  | 40.64 | 40.18 | 40.62       | 41.83        | 41.80   |
| **WR**   | 59.13  | 59.00 | 59.26 | 59.07       | 59.75        | 59.99   |

> The table shows how each model’s predictive accuracy varies by position.

---

## Table 7. Top LASSO Coefficients by Absolute Value

| Feature             | Coefficient   | Absolute Coefficient |
|---------------------|--------------:|----------------------:|
| completion_rate     | -17.8979890   | 17.8979890           |
| positionTE          | -10.8775789   | 10.8775789           |
| positionWR          |  -5.7420639   |  5.7420639           |
| over_10rush_game    |  -5.6925762   |  5.6925762           |
| interceptions       |  -2.9962910   |  2.9962910           |
| rushing_tds         |   2.7746831   |  2.7746831           |
| passing_tds         |   2.3370283   |  2.3370283           |
| receiving_tds       |   2.3233078   |  2.3233078           |
| positionRB          |  -1.2258538   |  1.2258538           |
| passing_efficiency  |   1.0386309   |  1.0386309           |
| rushing_efficiency  |   0.9226316   |  0.9226316           |
| fumbles_total       |  -0.5355879   |  0.5355879           |
| receptions          |   0.4428340   |  0.4428340           |
| catch_efficiency    |   0.4063188   |  0.4063188           |
| carries             |  -0.1842852   |  0.1842852           |
