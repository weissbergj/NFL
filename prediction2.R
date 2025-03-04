# Rscript prediction.r
###############################################################################
# ECON 108 Final Project
# 
# "Waking the Sleepers: Using Data Science to Discover Undervalued Players 
#  in PPR Scored Fantasy Football"
#
# Authors: Austin Bennett, Jared Weissberg
# 
# 1) Installs/Loads needed packages.
# 2) Loads NFL data (2010-2022) and aggregates by player-season.
# 3) Merges each season's stats with the FOLLOWING season's PPR => predictive.
# 4) Performs year-based splits (cross-validation by year) + final single test on 2022.
# 5) Fits:
#     (a) Linear Regression
#     (b) Lasso
#     (c) Ridge
#     (d) Random Forest
#     (e) XGBoost
#   ...and compares out-of-sample MSE for each approach.
# 6) Prints summary and chooses the best approach per fold.
# 
# THIS VERSION USES RICHER FEATURES THAN PREDICTION.R; IT ALSO PRINTS FEATURE IMPORTANCE in (6)
# 
###############################################################################

###########################
# 0) Install/Load Packages
###########################
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages_needed <- c(
  "nflreadr",
  "dplyr",
  "ggplot2",
  "glmnet",
  "randomForest",
  "xgboost",
  "Matrix"
)

for(p in packages_needed){
  if(!requireNamespace(p, quietly=TRUE))
    install.packages(p)
}

library(nflreadr)
library(dplyr)
library(ggplot2)
library(glmnet)        
library(randomForest)  
library(xgboost)       
library(Matrix)

###########################
# 1) Load & Prepare Data
###########################
cat("\n=== 1) LOADING MULTI-YEAR NFL DATA (2010-2022) ===\n")
df_raw <- load_player_stats(stat_type="offense", seasons=2010:2022)

cat("\n=== Aggregating Each Player's Season Stats ===\n")
df_season <- df_raw %>%
  group_by(player_id, player_display_name, position, season) %>%
  summarise(
    passing_yards   = sum(passing_yards,   na.rm=TRUE),
    passing_tds     = sum(passing_tds,     na.rm=TRUE),
    interceptions   = sum(interceptions,   na.rm=TRUE),
    carries         = sum(carries,         na.rm=TRUE),
    rushing_yards   = sum(rushing_yards,   na.rm=TRUE),
    rushing_tds     = sum(rushing_tds,     na.rm=TRUE),
    receptions      = sum(receptions,      na.rm=TRUE),
    receiving_yards = sum(receiving_yards, na.rm=TRUE),
    receiving_tds   = sum(receiving_tds,   na.rm=TRUE),
    fantasy_points_ppr = sum(fantasy_points_ppr, na.rm=TRUE),
    completions     = sum(completions,     na.rm=TRUE),
    pass_attempts   = sum(attempts,        na.rm=TRUE),
    fumbles_total   = sum(rushing_fumbles, na.rm=TRUE) + 
                      sum(sack_fumbles,    na.rm=TRUE) +
                      sum(receiving_fumbles, na.rm=TRUE),
    .groups="drop"
  ) %>%
  filter(position %in% c("QB","RB","WR","TE"))

cat("\n=== Creating Next-Season PPR Column (Predictive) ===\n")
df_next <- df_season %>%
  mutate(next_season = season + 1) %>%
  rename(ppr_this_season = fantasy_points_ppr)

df_predict <- df_next %>%
  inner_join(
    df_season %>%
      select(player_id, season, fantasy_points_ppr) %>%
      rename(
        ppr_next_season = fantasy_points_ppr,
        next_season_match = season
      ),
    by = c("player_id" = "player_id",
           "next_season" = "next_season_match")
  )

cat("\nHead of 'df_predict' dataset:\n")
print(head(df_predict))

df_predict <- df_predict %>%
  mutate(
    passing_efficiency = ifelse(position=="QB" & completions>0,
                                passing_yards / completions, 0),
    completion_rate    = ifelse(position=="QB" & pass_attempts>0,
                                completions / pass_attempts, 0),
    catch_efficiency   = ifelse(position %in% c("WR","TE") & receptions>0,
                                receiving_yards / receptions, 0),
    rushing_efficiency = ifelse(position=="RB" & carries>0,
                                rushing_yards / carries, 0),
    over_10rush_game   = ifelse(position=="RB" & carries>=160, 1, 0)
  )

make_model_matrix <- function(df){
  model.matrix(
    ppr_next_season ~ 
      passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds +
      position +
      completions + pass_attempts + fumbles_total +
      passing_efficiency + completion_rate +
      catch_efficiency + rushing_efficiency + over_10rush_game,
    data=df
  )[,-1]
}

###########################
# 2) Year-Based Cross-Validation
###########################
all_seasons <- sort(unique(df_predict$season))
valid_seasons <- all_seasons[ all_seasons <= 2021 ]

cat("\n=== 2) DEFINING CROSS-VALIDATION OVER YEARS ===\n")

fit_and_evaluate <- function(train_df, test_df){
  out <- list()
  
  lm_fit <- lm(
    ppr_next_season ~ passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds + position +
      completions + pass_attempts + fumbles_total +
      passing_efficiency + completion_rate +
      catch_efficiency + rushing_efficiency + over_10rush_game,
    data=train_df
  )
  pred_lm <- predict(lm_fit, newdata=test_df)
  mse_lm   <- mean((test_df$ppr_next_season - pred_lm)^2)
  out$mse_lm <- mse_lm
  
  x_train <- make_model_matrix(train_df)
  y_train <- train_df$ppr_next_season
  x_test  <- make_model_matrix(test_df)
  y_test  <- test_df$ppr_next_season
  
  set.seed(100)
  cv_lasso <- cv.glmnet(x_train, y_train, alpha=1, nfolds=5)
  pred_lasso <- predict(cv_lasso, newx=x_test, s="lambda.min")
  mse_lasso  <- mean((y_test - pred_lasso)^2)
  out$mse_lasso <- mse_lasso
  
  set.seed(101)
  cv_ridge <- cv.glmnet(x_train, y_train, alpha=0, nfolds=5)
  pred_ridge <- predict(cv_ridge, newx=x_test, s="lambda.min")
  mse_ridge  <- mean((y_test - pred_ridge)^2)
  out$mse_ridge <- mse_ridge
  
  set.seed(102)
  train_rf <- train_df %>% mutate(position = factor(position))
  test_rf  <- test_df %>% mutate(position = factor(position))
  rf_fit <- randomForest(
    ppr_next_season ~ passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds + position +
      completions + pass_attempts + fumbles_total +
      passing_efficiency + completion_rate +
      catch_efficiency + rushing_efficiency + over_10rush_game,
    data=train_rf,
    ntree=500,
    mtry=6,
    importance=FALSE
  )
  pred_rf <- predict(rf_fit, newdata=test_rf)
  mse_rf  <- mean((test_rf$ppr_next_season - pred_rf)^2)
  out$mse_rf <- mse_rf
  
  set.seed(103)
  xgb_dtrain <- xgb.DMatrix(data=x_train, label=y_train)
  xgb_dtest  <- xgb.DMatrix(data=x_test,  label=y_test)
  params <- list(
    objective="reg:squarederror",
    eta=0.08,
    max_depth=5
  )
  xgb_fit <- xgb.train(
    params=params,
    data=xgb_dtrain,
    nrounds=300,
    watchlist=list(train=xgb_dtrain),
    verbose=0
  )
  pred_xgb <- predict(xgb_fit, newdata=xgb_dtest)
  mse_xgb  <- mean((y_test - pred_xgb)^2)
  out$mse_xgb <- mse_xgb
  
  out
}

cv_results <- data.frame(
  test_year   = integer(),
  mse_lm      = numeric(),
  mse_lasso   = numeric(),
  mse_ridge   = numeric(),
  mse_rf      = numeric(),
  mse_xgb     = numeric(),
  best_model  = character(),
  stringsAsFactors=FALSE
)

cat("\n=== 3) RUNNING YEAR-BASED CROSS-VALIDATION SPLITS ===\n")
for(test_year in valid_seasons){
  train_df <- df_predict %>% filter(season < test_year)
  test_df  <- df_predict %>% filter(season == test_year)
  
  if(nrow(test_df) == 0 || nrow(train_df) == 0){
    next
  }
  
  cat("  -> FOLD w/ test_year =", test_year, 
      " (Train size =", nrow(train_df), ", Test size =", nrow(test_df), ")\n")
  
  out_list <- fit_and_evaluate(train_df, test_df)
  
  model_names <- c("lm","lasso","ridge","rf","xgb")
  mse_values  <- c(
    out_list$mse_lm, out_list$mse_lasso, out_list$mse_ridge,
    out_list$mse_rf, out_list$mse_xgb
  )
  best_idx <- which.min(mse_values)
  best_mod <- model_names[best_idx]
  
  cv_results <- rbind(cv_results, data.frame(
    test_year  = test_year,
    mse_lm     = out_list$mse_lm,
    mse_lasso  = out_list$mse_lasso,
    mse_ridge  = out_list$mse_ridge,
    mse_rf     = out_list$mse_rf,
    mse_xgb    = out_list$mse_xgb,
    best_model = best_mod,
    stringsAsFactors=FALSE
  ))
}

cat("\n=== CROSS-VALIDATION RESULTS (by test_year) ===\n")
print(cv_results)

cat("\n=== AVERAGE MSE ACROSS ALL FOLDS ===\n")
avg_mse <- cv_results %>%
  summarise(
    lm=mean(mse_lm),
    lasso=mean(mse_lasso),
    ridge=mean(mse_ridge),
    rf=mean(mse_rf),
    xgb=mean(mse_xgb)
  )
print(avg_mse)

cat("\nWhich model is best on average?\n")
print(t(avg_mse))

best_counts <- table(cv_results$best_model)
cat("\nBest Model Counts (lowest MSE per fold):\n")
print(best_counts)

###########################
# 4) Final "Train on All but 2021, Predict on 2021 -> 2022"
###########################
cat("\n=== 4) FINAL TRAIN on [2010..2020], TEST on 2021 => Next Season (2022) ===\n")
train_final <- df_predict %>% filter(season < 2021)
test_final  <- df_predict %>% filter(season == 2021)

final_out <- fit_and_evaluate(train_final, test_final)
print(final_out)

mse_values  <- c(final_out$mse_lm,
                 final_out$mse_lasso,
                 final_out$mse_ridge,
                 final_out$mse_rf,
                 final_out$mse_xgb)
model_names <- c("lm","lasso","ridge","rf","xgb")
best_idx <- which.min(mse_values)
cat("\nBest model for 2021->2022 was:", model_names[best_idx],
    "with MSE=", round(mse_values[best_idx],2), "\n")

###########################
# 5) Summaries
###########################
cat("\n\n=== COMPLETE SUMMARY OF RESULTS ===\n")
cat("Year-based CV results:\n")
print(cv_results)

cat("\nAverage MSE by model:\n")
print(avg_mse)

cat("\nSingle final test on 2021->2022:\n")
print(final_out)
cat("\nBest final model was:", model_names[best_idx], "\n")

###########################
# 6) Feature Importance
###########################
# Lasso-based ranking by coefficient magnitudes
cat("\n=== 6) FEATURE IMPORTANCE (LASSO) ===\n")
X_all <- make_model_matrix(df_predict)
y_all <- df_predict$ppr_next_season
set.seed(999)
cv_lasso_all <- cv.glmnet(X_all, y_all, alpha=1, nfolds=5)
best_lam <- cv_lasso_all$lambda.min
lasso_coefs <- coef(cv_lasso_all, s=best_lam)
lasso_df <- data.frame(
  feature = row.names(lasso_coefs),
  coef    = as.numeric(lasso_coefs)
)
lasso_df <- lasso_df[-1,] # drop intercept
lasso_df$abs_coef <- abs(lasso_df$coef)
lasso_df <- lasso_df[order(lasso_df$abs_coef, decreasing=TRUE),]
cat("\nTop LASSO Coefficients by Absolute Value:\n")
print(lasso_df[1:min(nrow(lasso_df), 15), ])

# Random Forest-based importance (re-fit with importance=TRUE)
cat("\n=== FEATURE IMPORTANCE (RANDOM FOREST) ===\n")
train_rf_import <- df_predict %>% mutate(position = factor(position))
set.seed(998)
rf_import_fit <- randomForest(
  ppr_next_season ~ passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds +
    receptions + receiving_yards + receiving_tds + position +
    completions + pass_attempts + fumbles_total +
    passing_efficiency + completion_rate +
    catch_efficiency + rushing_efficiency + over_10rush_game,
  data=train_rf_import,
  ntree=500,
  mtry=6,
  importance=TRUE
)
rf_imp <- importance(rf_import_fit)
cat("\nRandom Forest Variable Importance (MeanDecreaseGini):\n")
print(rf_imp)

cat("\n*** DONE ***\n")
