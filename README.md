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

## 1) Results from `output.log` (First Predictive Approach)

| Model         | Mean MSE (CV)* | Final Test MSE†   |
|---------------|----------------|--------------------|
| Linear        | 4698.505       | **4018.003**       |
| Lasso         | 4702.563       | 4021.675           |
| **Ridge**     | **4680.180**   | 4066.800           |
| RandomForest  | 4987.957       | 4241.713           |
| XGBoost       | 5575.036       | 4811.544           |

---

## 2) Results from `output2.log` (Second Predictive Approach)

| Model         | Mean MSE (CV)* | Final Test MSE†   |
|---------------|----------------|--------------------|
| Linear        | 4770.978       | 4051.395           |
| **Lasso**     | 4698.119       | **4049.582**       |
| **Ridge**     | **4681.676**   | 4050.882           |
| RandomForest  | 4992.539       | 4211.980           |
| XGBoost       | 5368.069       | 4471.895           |

> **Notes**  
> \* **Mean MSE (CV)** is the average mean squared error over 11 cross-validation folds (2011–2021).  
> † **Final Test MSE** is from training on 2010–2020 and testing on the 2021 season (predicting 2022 PPR).

## 3) Top LASSO Coefficients by Absolute Value

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
