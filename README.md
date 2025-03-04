# NFL Offensive Player Data: Analysis and Prediction

**Authors:** Austin Bennett and Jared Weissberg

In this repository, we analyze offensive player data in the NFL in R. Below is a brief description of each file:

- [**summary_stats.R**](https://github.com/weissbergj/NFL/blob/main/summary_stats.R)  
  Provides initial data visualizations and summary statistics for 2022 weekly data.

- [**summary_stats.log**](https://github.com/weissbergj/NFL/blob/main/summary_stats.log)  
  Contains the console output from running **summary_stats.R**.

- [**attempt1.R**](https://github.com/weissbergj/NFL/blob/main/attempt1.R)  
  An initial script experimenting with linear regressions and k-means clustering. It does not predict but carries over methods from earlier assignments.

- [**prediction.R**](https://github.com/weissbergj/NFL/blob/main/prediction.R)  
  A first attempt at predictive modeling using linear regression, LASSO, Ridge, random forest, and XGBoost. Results are logged in **output.log**.

- [**prediction2.R**](https://github.com/weissbergj/NFL/blob/main/prediction2.R)  
  Extends the predictive approach with additional features and computes feature importance. Results are logged in **output2.log**.

- [**output.log**](https://github.com/weissbergj/NFL/blob/main/output.log)  
  Contains the console output from running **prediction.R**.

- [**output2.log**](https://github.com/weissbergj/NFL/blob/main/output2.log)  
  Contains the console output from running **prediction2.R**.

- [**Rplots.pdf**](https://github.com/weissbergj/NFL/blob/main/Rplots.pdf)  
  Displays initial data visualizations, which are not highly informative.

## 1) Results from `output.log` (First Predictive Approach, **RMSE**)

| Model         | Mean RMSE (CV)* | Final Test RMSE† |
|---------------|-----------------|-------------------|
| Linear        | 68.57           | **63.42**        |
| Lasso         | 68.59           | 63.44            |
| **Ridge**     | **68.38**       | 63.77            |
| RandomForest  | 70.61           | 65.16            |
| XGBoost       | 74.66           | 69.38            |

---

## 2) Results from `output2.log` (Second Predictive Approach, **RMSE**)

| Model         | Mean RMSE (CV)* | Final Test RMSE† |
|---------------|-----------------|-------------------|
| Linear        | 69.09           | 63.64            |
| **Lasso**     | 68.55           | **63.63**        |
| **Ridge**     | **68.41**       | 63.64            |
| RandomForest  | 70.63           | 64.89            |
| XGBoost       | 73.29           | 66.87            |

> **Notes**  
> \* **Mean RMSE (CV)** is the root mean squared error averaged over 11 year-based folds (2011–2021).  
> † **Final Test RMSE** is from training on 2010–2020 and testing on the 2021 data (predicting 2022 PPR).

## 3) RMSE by Position (Second Predictive Approach on Final test Set (2021 → 2022))

| Position | Linear | Lasso | Ridge | RandomForest | XGBoost |
|----------|-------:|------:|------:|------------:|--------:|
| **QB**   | 80.93  | 81.23 | 80.95 | 84.96       | 93.72   |
| **RB**   | 73.09  | 72.83 | 73.01 | 73.31       | 73.43   |
| **TE**   | 40.59  | 40.64 | 40.18 | 41.83       | 41.80   |
| **WR**   | 59.13  | 59.00 | 59.26 | 59.75       | 59.99   |

> The table shows how each model’s predictive accuracy varies by position. For instance, while Lasso narrowly edges out other methods for running backs, Ridge achieves the lowest RMSE for tight ends.

## 4) Top LASSO Coefficients by Absolute Value

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
