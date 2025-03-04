###############################################################################
# Comprehensive "Position-Specific" Modeling for Fantasy Football (PPR)
#
# This script:
#   1) Loads multi-year NFL data (via nflreadr) for 2010-2022
#   2) Adds advanced features (placeholders for injuries, coaching, etc.)
#   3) Splits data by position (QB, RB, WR, TE)
#   4) Performs hyperparameter tuning & cross-validation for:
#        (a) Linear (LM, Lasso, Ridge, Elastic)
#        (b) Random Forest (with grid search)
#        (c) XGBoost (with grid search)
#   5) Evaluates each position's best model(s), optionally stacking them
#   6) Predicts on final holdout season (train on 2010-2020, test on 2021)
#   7) Ranks combined predictions to produce top-10 (or standard roster)
#   8) Compares to actual outcomes
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

cat("\n=== Adding Additional / Advanced Features (Placeholders) ===\n")
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
    over_10rush_game   = ifelse(position=="RB" & carries>=160, 1, 0),
  )

#############################
# 1a) Splitting By Position
#############################
df_qb <- df_predict %>% filter(position=="QB")
df_rb <- df_predict %>% filter(position=="RB")
df_wr <- df_predict %>% filter(position=="WR")
df_te <- df_predict %>% filter(position=="TE")

#############################
# 1b) Helper to define "X" matrix
#############################
# because each position might have different features, we define a helper
# that returns a model matrix for each position specifically:
make_position_matrix <- function(df, pos){
  if(pos=="QB"){
    form <- as.formula("ppr_next_season ~ 
      passing_yards + passing_tds + interceptions +
      completions + pass_attempts + 
      rushing_yards + rushing_tds + 
      fumbles_total + 
      passing_efficiency + completion_rate
      # + placeholders for advanced or external data
    ")
  } else if(pos=="RB"){
    form <- as.formula("ppr_next_season ~ 
      carries + rushing_yards + rushing_tds +
      receptions + receiving_yards + receiving_tds +
      fumbles_total +
      rushing_efficiency + over_10rush_game
      # + placeholders for advanced data
    ")
  } else if(pos=="WR"){
    form <- as.formula("ppr_next_season ~
      receptions + receiving_yards + receiving_tds +
      rushing_yards + rushing_tds +
      fumbles_total +
      catch_efficiency
      # + placeholders for advanced data
    ")
  } else if(pos=="TE"){
    form <- as.formula("ppr_next_season ~
      receptions + receiving_yards + receiving_tds +
      rushing_tds + rushing_yards +
      fumbles_total +
      catch_efficiency
      # + placeholders for advanced data
    ")
  } else {
    stop("Unknown position in make_position_matrix()!")
  }

  model.matrix(form, data=df)[, -1, drop=FALSE]  # drop intercept
}


###############################################################
# 2) CROSS-VALIDATION & HYPERPARAM TUNING (Per Position)
###############################################################
# define a function that:
#   1) accepts a training set, a test set, and a position label
#   2) builds LM, Lasso, Ridge, Elastic Net
#   3) builds Random Forest & XGBoost with hyperparam search
#   4) returns the final predictions & MSE

library(stats)  # for lm
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
  # build a formula by re-using the same logic as in "make_position_matrix"
  # or can do direct use of model.matrix, then do "lm.fit()"
  # for brevity, we do a "dummy formula" approach:

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
  best_lasso <- cv_lasso$lambda.min
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
  # quick search over alpha in {0.25, 0.5, 0.75}
  # pick the best alpha by CV. This is a mini grid approach:
  possible_alphas <- c(0.25, 0.5, 0.75)
  best_elastic_mse <- Inf
  best_elastic_fit <- NULL
  best_elastic_alpha <- NA
  for(a in possible_alphas){
    set.seed(125 + round(100*a))
    cv_en <- cv.glmnet(x_train, y_train, alpha=a, nfolds=5)
    if(cv_en$cvm[cv_en$lambda == cv_en$lambda.min] < best_elastic_mse){
      best_elastic_mse <- cv_en$cvm[cv_en$lambda == cv_en$lambda.min]
      best_elastic_fit <- cv_en
      best_elastic_alpha <- a
    }
  }
  pred_elastic <- predict(best_elastic_fit, newx=x_test, s="lambda.min")
  mse_elastic <- mean((y_test - pred_elastic)^2)
  out$mse_elastic <- mse_elastic
  out$pred_elastic <- as.vector(pred_elastic)
  out$best_en_alpha <- best_elastic_alpha

  #################
  # (e) Random Forest (Hyperparam Tuning)
  #################
  rf_grid <- expand.grid(
    ntree = c(200, 500),
    mtry  = c(2, 4, 6, 8)  # depends on features
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
  out$best_rf_fit <- best_rf_fit

  #################
  # (f) XGBoost (Hyperparam Tuning)
  #################
  xgb_grid <- expand.grid(
    eta = c(0.02, 0.08, 0.15),
    max_depth = c(3, 5, 7)
  )
  best_xgb_mse <- Inf
  best_xgb_model <- NULL

  dtrain <- xgb.DMatrix(data=x_train, label=y_train)
  dtest  <- xgb.DMatrix(data=x_test,  label=y_test)

  for(i in seq_len(nrow(xgb_grid))){
    curr_eta <- xgb_grid$eta[i]
    curr_md  <- xgb_grid$max_depth[i]
    
    xgb_params <- list(
      objective="reg:squarederror",
      eta=curr_eta,
      max_depth=curr_md
    )
    set.seed(300 + i)
    xgb_model_temp <- xgb.train(
      params=xgb_params,
      data=dtrain,
      nrounds=500,  # could tune or do early stopping
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
  out$best_xgb_model <- best_xgb_model

  return(out)
}


##########################################
# 3) Master "Train -> Test" for each Pos
##########################################
# Define a function that given:
#   - train_df, test_df for a specific position
#   - calls position_fit_and_evaluate()
#   - identifies best model, returns everything

evaluate_position <- function(train_df, test_df, pos){
  cat("\n--- Evaluating Position:", pos, "---\n")
  results <- position_fit_and_evaluate(train_df, test_df, pos)
  
  # pick best model by MSE:
  model_names <- c("lm","lasso","ridge","elastic","rf","xgb")
  mse_vals <- c(results$mse_lm, results$mse_lasso, results$mse_ridge,
                results$mse_elastic, results$mse_rf, results$mse_xgb)
  best_idx <- which.min(mse_vals)
  best_model <- model_names[best_idx]

  cat("Position:", pos, " Best model is:", best_model, 
      " with MSE=", round(min(mse_vals),2), "\n")
  
  results$best_model <- best_model
  return(results)
}

##########################################
# 4) "Year-based" CV approach (by position)
##########################################
# Instead of combining all positions for year-based CV,
# do a position-level approach. 
# define valid seasons for each pos, etc.

cv_results_all <- list(QB=data.frame(), RB=data.frame(), WR=data.frame(), TE=data.frame())

all_seasons <- sort(unique(df_predict$season))
valid_seasons <- all_seasons[ all_seasons <= 2021 ]
pos_list <- c("QB","RB","WR","TE")

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

for(pos in pos_list){
  cat("\n=== CV Results for", pos, "===\n")
  print(cv_results_all[[pos]])
}

###############################################################
# 5) Final Train on [2010..2020], Test on 2021 -> 2022
###############################################################
cat("\n=== FINAL TRAIN [2010..2020], TEST on 2021, per position ===\n")

df_train_final <- df_predict %>% filter(season < 2021)
df_test_final  <- df_predict %>% filter(season == 2021)

# position-level approach
final_preds_list <- list()

for(pos in pos_list){
  train_pos <- df_train_final %>% filter(position==pos)
  test_pos  <- df_test_final %>% filter(position==pos)
  
  if(nrow(test_pos)==0 || nrow(train_pos)==0){
    # no players or something
    next
  }
  
  cat("\n--- Final Train/Pred for pos =", pos, " ---\n")
  out_final <- position_fit_and_evaluate(train_pos, test_pos, pos)
  
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
  cat("BEST model for", pos, "is", best_mod, "with MSE =", round(min(mse_values),2), "\n")

  # store best model's predictions
  best_preds <- switch(
    best_mod,
    "lm"      = out_final$pred_lm,
    "lasso"   = out_final$pred_lasso,
    "ridge"   = out_final$pred_ridge,
    "elastic" = out_final$pred_elastic,
    "rf"      = out_final$pred_rf,
    "xgb"     = out_final$pred_xgb
  )
  
  # Attach predictions to the test_pos data
  test_pos$pred_position_ppr <- best_preds
  test_pos$best_model        <- best_mod
  
  final_preds_list[[pos]] <- test_pos
}

##################################################
# 6) Combine All Positions & Create a Single Rank
##################################################
cat("\n=== COMBINING PREDICTIONS FOR ALL POSITIONS ===\n")
final_pred_df <- do.call(rbind, final_preds_list)

# rename "pred_position_ppr" -> "predicted_ppr"
colnames(final_pred_df)[which(colnames(final_pred_df)=="pred_position_ppr")] <- "predicted_ppr"

# Now we can rank across all players:
final_pred_df <- final_pred_df %>%
  mutate(
    pred_rank_all  = min_rank(desc(predicted_ppr)),
    actual_rank_all= min_rank(desc(ppr_next_season))
  )

##################################################
# 7) Generate a Final "Top 10" (Ignoring Positions)
##################################################
cat("\n=== FINAL TOP 10 (ANY POSITION, BY predicted_ppr) ===\n")
top10_anypos <- final_pred_df %>%
  filter(pred_rank_all <= 10) %>%
  arrange(pred_rank_all)

print(top10_anypos %>%
  select(pred_rank_all, player_display_name, position, ppr_next_season, predicted_ppr))

cat("\n=== ACTUAL TOP 10 (ANY POSITION) ===\n")
top10_actual_anypos <- final_pred_df %>%
  filter(actual_rank_all <= 10) %>%
  arrange(actual_rank_all)

print(top10_actual_anypos %>%
  select(actual_rank_all, player_display_name, position, ppr_next_season, predicted_ppr))

# Could compute overlap, difference, etc.
overlap_ids <- intersect(top10_anypos$player_id, top10_actual_anypos$player_id)
cat("\nOverlap in top-10 any-position:", length(overlap_ids), "players\n")

# Summaries
sum_top10_pred <- sum(top10_anypos$ppr_next_season, na.rm=TRUE)
sum_top10_actual <- sum(top10_actual_anypos$ppr_next_season, na.rm=TRUE)
cat("\nPoints by predicted top-10:", round(sum_top10_pred,1),
    "\nPoints by actual top-10:", round(sum_top10_actual,1),
    "\nDifference:", round(sum_top10_actual - sum_top10_pred,1), "\n")

##################################################
# 8) Also Evaluate a Standard Roster
##################################################
cat("\n=== DRAFTING A STANDARD ROSTER (1 QB, 2 RB, 2 WR, 1 TE) BY predicted_ppr ===\n")

pos_constraints <- c(QB=1, RB=2, WR=2, TE=1)
team_picks <- list()
for(p in names(pos_constraints)){
  needed <- pos_constraints[p]
  subdf <- final_pred_df %>%
    filter(position==p) %>%
    arrange(desc(predicted_ppr))
  picks <- head(subdf, needed)
  team_picks[[p]] <- picks
}
team_final <- do.call(rbind, team_picks)
team_score <- sum(team_final$ppr_next_season, na.rm=TRUE)
cat("Model-based standard roster score:", round(team_score,1), "\n")

# Compare to "best actual roster" (by ppr_next_season):
best_team_picks <- list()
for(p in names(pos_constraints)){
  needed <- pos_constraints[p]
  subdf <- final_pred_df %>%
    filter(position==p) %>%
    arrange(desc(ppr_next_season))
  picks <- head(subdf, needed)
  best_team_picks[[p]] <- picks
}
best_team_final <- do.call(rbind, best_team_picks)
best_team_score <- sum(best_team_final$ppr_next_season, na.rm=TRUE)

cat("Actual best possible roster (same constraints) would have scored:",
    round(best_team_score,1), "\n")
cat("Difference:", round(best_team_score - team_score,1), "\n")

##################################################
# 9) Done
##################################################
cat("\n*** DONE ***\n")
