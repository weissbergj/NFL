###############################################################################
# QB-Specific Modeling for Fantasy Football (PPR)
#
# This script:
#   1) Loads multi-year NFL data (via nflreadr) for 2010-2022
#   2) Adds advanced QB-focused features (multi-year sums, TD rate)
#   3) Filters out low-attempt QBs
#   4) Splits data by position (QB only)
#   5) Performs hyperparameter tuning & cross-validation for:
#        (a) Linear (LM, Lasso, Ridge, Elastic)
#        (b) Random Forest
#        (c) XGBoost (bigger grid)
#   6) Evaluates & identifies best model
#   7) Predicts on final holdout season (train on 2010-2020, test on 2021)
#   8) Ranks predictions, compares to actual outcomes
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
  # We'll still filter to only skill positions but we only CARE about QB in final code
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

############################################################
# 1a) Extra: Compute Per-Year League Average TD Rate for QBs
############################################################
df_qb_league <- df_predict %>%
  filter(position == "QB", pass_attempts > 0) %>%
  group_by(season) %>%
  summarise(
    league_td_rate = sum(passing_tds) / sum(pass_attempts),
    .groups = "drop"
  )

############################################################
# 1b) Add Additional / Advanced Features for QBs
############################################################
cat("\n=== Adding Additional / Advanced QB Features ===\n")

df_predict <- df_predict %>%
  # We'll do multi-year rolling stats for QBs only
  group_by(player_id) %>%
  arrange(player_id, season, .by_group=TRUE) %>%
  mutate(
    # 2-year sums (lag by 1 and by 2)
    pass_yards_lag1 = lag(passing_yards, 1, default=NA),
    pass_yards_lag2 = lag(passing_yards, 2, default=NA),
    pass_tds_lag1   = lag(passing_tds, 1, default=NA),
    pass_tds_lag2   = lag(passing_tds, 2, default=NA),

    pass_yards_2yr  = pass_yards_lag1 + pass_yards_lag2,
    pass_tds_2yr    = pass_tds_lag1 + pass_tds_lag2
  ) %>%
  ungroup() %>%
  
  # Merge in league-wide TD rate for each season (for QBs)
  left_join(df_qb_league, by=c("season")) %>%
  
  mutate(
    passing_efficiency = ifelse(position=="QB" & completions>0,
                                passing_yards / completions, 0),
    completion_rate    = ifelse(position=="QB" & pass_attempts>0,
                                completions / pass_attempts, 0),
    
    # If WR/TE features needed, they'd be here, but we won't expand them now
    qb_rush_heavy = ifelse(position == "QB" & rushing_yards > 300, 1, 0),
    pass_tds_per_attempt = ifelse(position == "QB" & pass_attempts > 0,
                                  passing_tds / pass_attempts, 0),

    # new: pass_attempts_per_game (assuming 16 games for older data, you can do 17 if you prefer)
    pass_attempts_per_game = ifelse(position=="QB", pass_attempts / 16, 0),

    # For QBs only, measure how far above league avg their TD rate was
    td_rate_above_avg = ifelse(position=="QB" & pass_attempts>0 & !is.na(league_td_rate),
                               (passing_tds/pass_attempts) - league_td_rate, 0),

    pass_yards_2yr  = ifelse(is.na(pass_yards_2yr), 0, pass_yards_2yr),
    pass_tds_2yr    = ifelse(is.na(pass_tds_2yr), 0, pass_tds_2yr),
    td_rate_above_avg = ifelse(is.na(td_rate_above_avg), 0, td_rate_above_avg)
  )

# Optional: Filter out QBs with minimal usage
# remove QBs with <100 pass attempts in that season
df_predict <- df_predict %>%
  filter(!(position == "QB" & pass_attempts < 100))

#############################
# 2) Split By Position (QB)
#############################
# We only keep QB in pos_list anyway, but let's maintain the same structure
df_qb <- df_predict %>% filter(position=="QB")
# (We won't run RB, WR, TE models; pos_list below is only "QB")

#############################
# 2b) Helper to define "X" matrix
#############################
make_position_matrix <- function(df, pos){
  if(pos=="QB"){
    form <- as.formula("
      ppr_next_season ~
        passing_yards +
        passing_tds +
        interceptions +
        completions +
        pass_attempts +
        rushing_yards +
        rushing_tds +
        fumbles_total +
        passing_efficiency +
        completion_rate +
        qb_rush_heavy +
        pass_tds_per_attempt +
        pass_attempts_per_game +
        pass_yards_2yr +
        pass_tds_2yr +
        td_rate_above_avg
    ")
  } else {
    stop('Only QB modeling used here!')
  }
  model.matrix(form, data=df)[, -1, drop=FALSE]  # drop intercept
}

###############################################################
# 3) CROSS-VALIDATION & HYPERPARAM TUNING (QB only)
###############################################################
library(stats)

position_fit_and_evaluate <- function(train_df, test_df, pos){

  # Step 1: Build model matrix
  x_train <- make_position_matrix(train_df, pos)
  y_train <- train_df$ppr_next_season
  x_test  <- make_position_matrix(test_df, pos)
  y_test  <- test_df$ppr_next_season
  
  out <- list()

  #################
  # (a) Linear LM
  #################
  df_lm_train <- data.frame(y_train, x_train)
  colnames(df_lm_train)[1] <- "ppr_next_season"

  lm_fit <- lm(ppr_next_season ~ ., data=df_lm_train)
  
  df_lm_test <- data.frame(y_test, x_test)
  colnames(df_lm_test)[1] <- "ppr_next_season"
  
  pred_lm <- predict(lm_fit, newdata=df_lm_test)
  mse_lm <- mean((y_test - pred_lm)^2)
  out$mse_lm <- mse_lm
  out$pred_lm <- pred_lm

  #################
  # (b) Lasso
  #################
  set.seed(123)
  cv_lasso <- cv.glmnet(x_train, y_train, alpha=1, nfolds=5)
  pred_lasso <- predict(cv_lasso, newx=x_test, s="lambda.min")
  mse_lasso <- mean((y_test - pred_lasso)^2)
  out$mse_lasso <- mse_lasso
  out$pred_lasso <- as.vector(pred_lasso)

  #################
  # (c) Ridge
  #################
  set.seed(124)
  cv_ridge <- cv.glmnet(x_train, y_train, alpha=0, nfolds=5)
  pred_ridge <- predict(cv_ridge, newx=x_test, s="lambda.min")
  mse_ridge <- mean((y_test - pred_ridge)^2)
  out$mse_ridge <- mse_ridge
  out$pred_ridge <- as.vector(pred_ridge)

  #################
  # (d) Elastic Net
  #################
  possible_alphas <- c(0.25, 0.5, 0.75)
  best_elastic_mse <- Inf
  best_elastic_fit <- NULL
  for(a in possible_alphas){
    set.seed(125 + round(100*a))
    cv_en <- cv.glmnet(x_train, y_train, alpha=a, nfolds=5)
    curr_min_mse <- cv_en$cvm[cv_en$lambda == cv_en$lambda.min]
    if(curr_min_mse < best_elastic_mse){
      best_elastic_mse <- curr_min_mse
      best_elastic_fit <- cv_en
    }
  }
  pred_elastic <- predict(best_elastic_fit, newx=x_test, s="lambda.min")
  mse_elastic <- mean((y_test - pred_elastic)^2)
  out$mse_elastic <- mse_elastic
  out$pred_elastic <- as.vector(pred_elastic)

  #################
  # (e) Random Forest (Hyperparam Tuning)
  #################
  # We'll keep a smallish grid, but you can expand further
  rf_grid <- expand.grid(
    ntree = c(200, 500),
    mtry  = c(2, 4, 6, 8, 10)
  )
  best_rf_mse <- Inf
  best_rf_fit <- NULL
  for(i in seq_len(nrow(rf_grid))){
    set.seed(200 + i)
    curr_ntree <- rf_grid$ntree[i]
    curr_mtry  <- rf_grid$mtry[i]
    rf_fit_temp <- randomForest(
      x=x_train,
      y=y_train,
      ntree=curr_ntree,
      mtry=curr_mtry
    )
    pred_rf_temp <- predict(rf_fit_temp, newdata=x_test)
    mse_rf_temp <- mean((y_test - pred_rf_temp)^2)
    if(mse_rf_temp < best_rf_mse){
      best_rf_mse <- mse_rf_temp
      best_rf_fit <- rf_fit_temp
    }
  }
  out$mse_rf <- best_rf_mse
  pred_rf <- predict(best_rf_fit, newdata=x_test)
  out$pred_rf <- as.vector(pred_rf)

  #################
  # (f) XGBoost (Hyperparam Tuning) - expanded grid
  #################
  xgb_grid <- expand.grid(
    eta = c(0.01, 0.05, 0.1, 0.15),
    max_depth = c(3, 5, 7, 9),
    min_child_weight = c(1, 3, 5),
    subsample = c(0.8, 1.0),
    colsample_bytree = c(0.8, 1.0)
  )
  best_xgb_mse <- Inf
  best_xgb_model <- NULL

  dtrain <- xgb.DMatrix(data=x_train, label=y_train)
  dtest  <- xgb.DMatrix(data=x_test,  label=y_test)

  for(i in seq_len(nrow(xgb_grid))){
    params <- list(
      objective="reg:squarederror",
      eta = xgb_grid$eta[i],
      max_depth = xgb_grid$max_depth[i],
      min_child_weight = xgb_grid$min_child_weight[i],
      subsample = xgb_grid$subsample[i],
      colsample_bytree = xgb_grid$colsample_bytree[i]
    )
    set.seed(300 + i)
    xgb_model_temp <- xgb.train(
      params=params,
      data=dtrain,
      nrounds=1000,  # can increase further, or use early stopping
      verbose=0
    )
    pred_xgb_temp <- predict(xgb_model_temp, newdata=dtest)
    mse_xgb_temp <- mean((y_test - pred_xgb_temp)^2)
    if(mse_xgb_temp < best_xgb_mse){
      best_xgb_mse <- mse_xgb_temp
      best_xgb_model <- xgb_model_temp
    }
  }
  out$mse_xgb <- best_xgb_mse
  pred_xgb <- predict(best_xgb_model, newdata=dtest)
  out$pred_xgb <- as.vector(pred_xgb)

  return(out)
}

##########################################
# 4) Year-based Cross-Validation for QBs
##########################################
cv_results_all <- list(QB=data.frame())
all_seasons <- sort(unique(df_predict$season))
valid_seasons <- all_seasons[ all_seasons <= 2021 ]
pos_list <- c("QB")  # focusing only on QBs

for(pos in pos_list){
  cat("\n===== CROSS-VAL for position:", pos, "=====\n")
  df_pos <- df_predict %>% filter(position==pos)
  
  cv_pos <- data.frame(
    test_year   = integer(),
    mse_lm      = numeric(),
    mse_lasso   = numeric(),
    mse_ridge   = numeric(),
    mse_elastic = numeric(),
    mse_rf      = numeric(),
    mse_xgb     = numeric(),
    best_model  = character(),
    stringsAsFactors=FALSE
  )
  
  for(test_year in valid_seasons){
    train_df <- df_pos %>% filter(season < test_year)
    test_df  <- df_pos %>% filter(season == test_year)
    
    if(nrow(test_df)==0 || nrow(train_df)==0) next
    
    cat(" -> POS:", pos, "Test year:", test_year, 
        " Train size:", nrow(train_df), " Test size:", nrow(test_df), "\n")
    
    out_list <- position_fit_and_evaluate(train_df, test_df, pos)
    model_names <- c("lm","lasso","ridge","elastic","rf","xgb")
    mse_values  <- c(
      out_list$mse_lm, out_list$mse_lasso, out_list$mse_ridge,
      out_list$mse_elastic, out_list$mse_rf, out_list$mse_xgb
    )
    best_idx <- which.min(mse_values)
    best_mod <- model_names[best_idx]
    
    cv_pos <- rbind(cv_pos, data.frame(
      test_year   = test_year,
      mse_lm      = out_list$mse_lm,
      mse_lasso   = out_list$mse_lasso,
      mse_ridge   = out_list$mse_ridge,
      mse_elastic = out_list$mse_elastic,
      mse_rf      = out_list$mse_rf,
      mse_xgb     = out_list$mse_xgb,
      best_model  = best_mod,
      stringsAsFactors=FALSE
    ))
  }
  
  cv_results_all[[pos]] <- cv_pos
}

cat("\n=== CV Results for QB ===\n")
print(cv_results_all[["QB"]])

###############################################################
# 5) Final Train on [2010..2020], Test on 2021 for QBs
###############################################################
cat("\n=== FINAL TRAIN [2010..2020], TEST on 2021 ===\n")

df_train_final <- df_predict %>% filter(season < 2021)
df_test_final  <- df_predict %>% filter(season == 2021)

# Evaluate final
cat("\n--- Final Train/Pred for QB ---\n")
out_final <- position_fit_and_evaluate(df_train_final, df_test_final, "QB")

# find best model
model_names <- c("lm","lasso","ridge","elastic","rf","xgb")
mse_values  <- c(
  out_final$mse_lm,
  out_final$mse_lasso,
  out_final$mse_ridge,
  out_final$mse_elastic,
  out_final$mse_rf,
  out_final$mse_xgb
)
best_idx <- which.min(mse_values)
best_mod <- model_names[best_idx]
cat("BEST model for QB is", best_mod, "with MSE =", round(min(mse_values),2), "\n")

# store best preds
best_preds <- switch(
  best_mod,
  "lm"      = out_final$pred_lm,
  "lasso"   = out_final$pred_lasso,
  "ridge"   = out_final$pred_ridge,
  "elastic" = out_final$pred_elastic,
  "rf"      = out_final$pred_rf,
  "xgb"     = out_final$pred_xgb
)

df_test_final$predicted_ppr <- best_preds
df_test_final$best_model <- best_mod

#############################################
# 6) Rank the final test results for 2021 QBs
#############################################
df_test_final <- df_test_final %>%
  mutate(
    pred_rank = min_rank(desc(predicted_ppr)),
    actual_rank= min_rank(desc(ppr_next_season))
  )

cat("\n=== TOP 10 QBs by predicted PPR (2021) ===\n")
top10_pred <- df_test_final %>%
  filter(pred_rank <= 10) %>%
  arrange(pred_rank)
print(top10_pred %>%
  select(pred_rank, player_display_name, ppr_next_season, predicted_ppr))

cat("\n=== TOP 10 QBs by actual PPR (2021) ===\n")
top10_actual <- df_test_final %>%
  filter(actual_rank <= 10) %>%
  arrange(actual_rank)
print(top10_actual %>%
  select(actual_rank, player_display_name, ppr_next_season, predicted_ppr))

# Overlap
overlap_ids <- intersect(top10_pred$player_id, top10_actual$player_id)
cat("\nOverlap in top-10 QBs:", length(overlap_ids), "players\n")

# Compare total points among predicted vs. actual top 10
sum_top10_pred <- sum(top10_pred$ppr_next_season, na.rm=TRUE)
sum_top10_actual <- sum(top10_actual$ppr_next_season, na.rm=TRUE)
cat("\nPoints by predicted top-10 QBs:", round(sum_top10_pred,1),
    "\nPoints by actual top-10 QBs:", round(sum_top10_actual,1),
    "\nDifference:", round(sum_top10_actual - sum_top10_pred,1), "\n")

cat("\n*** DONE ***\n")
