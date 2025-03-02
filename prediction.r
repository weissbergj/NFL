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
# can modify or add advanced features, hyperparameter tuning, or classification tasks in future
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
library(glmnet)         # Lasso/Ridge
library(randomForest)   # RF
library(xgboost)        # XGB
library(Matrix)         # for sparse matrices used by xgboost

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
    # this is "this-year" PPR total:
    fantasy_points_ppr = sum(fantasy_points_ppr, na.rm=TRUE),
    .groups="drop"
  ) %>%
  # filter to main positions only:
  filter(position %in% c("QB","RB","WR","TE"))

cat("\n=== Creating Next-Season PPR Column (Predictive) ===\n")
# next-season approach:  merge "this season" stats with next season's PPR
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

# define a function to create a model matrix for Lasso/Ridge/XGBoost
# (i.e. ignoring the intercept).
make_model_matrix <- function(df){
  # remove columns not needed or that won't be numeric:
  # position might be turned into dummies (CAN ADJUST)
  # remove the key outcome columns
  # return matrix that fits glmnet/xgboost
  model.matrix(
    ppr_next_season ~ passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds +
      position,  # if you'd like position as a factor
    data=df
  )[,-1]  # drop the intercept column
}

###########################
# 2) Year-Based Cross-Validation
###########################
# define all "seasons" in df_predict that are valid (i.e. next_season up to 2022).
#  earliest season in df_predict is 2010, but that predicts 2011, etc.
# "leave out" each final year in [2015..2022], training on prior data.

all_seasons <- sort(unique(df_predict$season))
#  only consider splits where next_season <= 2022
# so that we have actual next_season data
valid_seasons <- all_seasons[ all_seasons <= 2021 ]  # 2021 predicts 2022

# define a function that, for each "test_year", trains on all seasons < test_year
# and tests on that "test_year" => next season is test_year+1.
# store MSE for each model type.
cat("\n=== 2) DEFINING CROSS-VALIDATION OVER YEARS ===\n")

# Helper function: fit all models & compute MSE
fit_and_evaluate <- function(train_df, test_df){
  # store results in named list
  out <- list()
  
  # (A) Linear Model
  lm_fit <- lm(
    ppr_next_season ~ passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds + position,
    data=train_df
  )
  pred_lm <- predict(lm_fit, newdata=test_df)
  mse_lm   <- mean((test_df$ppr_next_season - pred_lm)^2)
  out$mse_lm <- mse_lm
  
  # Prepare model matrix for Lasso/Ridge/XGBoost
  x_train <- make_model_matrix(train_df)
  y_train <- train_df$ppr_next_season
  x_test  <- make_model_matrix(test_df)
  y_test  <- test_df$ppr_next_season
  
  # (B) Lasso
  set.seed(100)
  cv_lasso <- cv.glmnet(x_train, y_train, alpha=1, nfolds=5)
  best_lam_lasso <- cv_lasso$lambda.min
  pred_lasso <- predict(cv_lasso, newx=x_test, s="lambda.min")
  mse_lasso  <- mean((y_test - pred_lasso)^2)
  out$mse_lasso <- mse_lasso
  
  # (C) Ridge
  set.seed(101)
  cv_ridge <- cv.glmnet(x_train, y_train, alpha=0, nfolds=5)
  best_lam_ridge <- cv_ridge$lambda.min
  pred_ridge <- predict(cv_ridge, newx=x_test, s="lambda.min")
  mse_ridge  <- mean((y_test - pred_ridge)^2)
  out$mse_ridge <- mse_ridge
  
  # (D) Random Forest
  set.seed(102)
  # must remove non-numerics or transform them for RF
  # recast 'position' as factor for the randomForest
  train_rf <- train_df %>%
    mutate(position = factor(position))
  test_rf  <- test_df %>%
    mutate(position = factor(position))  # ensure same factor levels
  
  rf_fit <- randomForest(
    ppr_next_season ~ passing_yards + passing_tds + interceptions +
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds + position,
    data=train_rf,
    ntree=300,
    importance=FALSE
  )
  pred_rf <- predict(rf_fit, newdata=test_rf)
  mse_rf  <- mean((test_rf$ppr_next_season - pred_rf)^2)
  out$mse_rf <- mse_rf
  
  # (E) XGBoost
  # do a quick example with default hyperparameters; tune nrounds, max_depth, etc. in future
  set.seed(103)
  xgb_dtrain <- xgb.DMatrix(data=x_train, label=y_train)
  xgb_dtest  <- xgb.DMatrix(data=x_test,  label=y_test)
  
  # Minimal parameter set; can tune these
  params <- list(
    objective="reg:squarederror",
    eta=0.1,        # learning rate
    max_depth=6
  )
  # 200 rounds
  xgb_fit <- xgb.train(
    params=params,
    data=xgb_dtrain,
    nrounds=200,
    watchlist=list(
      train=xgb_dtrain
      # ( could also add test=xgb_dtest to watch performance)
    ),
    verbose=0
  )
  pred_xgb <- predict(xgb_fit, newdata=xgb_dtest)
  mse_xgb  <- mean((y_test - pred_xgb)^2)
  out$mse_xgb <- mse_xgb
  
  # Return a named list with all MSE
  return(out)
}

#  store cross-validation results:
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
  # train: all seasons < test_year
  # test: season == test_year
  # note that the outcome is from next_season (test_year+1),
  # but we have that in the row "season==test_year" after merging.
  train_df <- df_predict %>%
    filter(season < test_year)
  test_df  <- df_predict %>%
    filter(season == test_year)
  
  if(nrow(test_df) == 0 || nrow(train_df) == 0){
    next
  }
  
  cat("  -> FOLD w/ test_year =", test_year, 
      " (Train size =", nrow(train_df), ", Test size =", nrow(test_df), ")\n")
  
  # fit models and get MSE
  out_list <- fit_and_evaluate(train_df, test_df)
  
  # which model is best for this fold?
  model_names <- c("lm","lasso","ridge","rf","xgb")
  mse_values  <- c(
    out_list$mse_lm, out_list$mse_lasso, out_list$mse_ridge,
    out_list$mse_rf, out_list$mse_xgb
  )
  best_idx <- which.min(mse_values)
  best_mod <- model_names[best_idx]
  
  # Store
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

#  can also see how often each model is "best" across the folds
best_counts <- table(cv_results$best_model)
cat("\nBest Model Counts (lowest MSE per fold):\n")
print(best_counts)

###########################
# 4) Final "Train on All but 2021, Predict on 2021 -> 2022"
###########################
# a final single train/test for the most recent season (2021 -> 2022).
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

cat("\n*** DONE ***\n")
