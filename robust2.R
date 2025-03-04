###############################################################################
# robust2.R
#
# 1) Loads main NFL offense data (2010-2022)
# 2) Loads multiple external data sources:
#    - injury logs
#    - coaching changes
#    - offensive line ranks
#    - advanced usage stats (snap share, red zone usage, target share, etc.)
# 3) Merges them into a single dataset by (player_id, season)
# 4) Splits into position-specific data (QB, RB, WR, TE)
# 5) Builds feature-rich design matrices for each position
# 6) Implements hyperparameter tuning for:
#     - Lasso, Ridge, Elastic Net (vary alpha, etc.)
#     - Random Forest (grid for ntree, mtry, nodesize)
#     - XGBoost (grid for eta, max_depth, subsample, colsample_bytree)
# 7) Trains each method per position with cross-validation
# 8) Stacks (ensembles) top 2-3 best models for each position
# 9) Does final train on [2010..2020], test on 2021
#10) Combines final predictions across positions, 
#    ranks for a top-10 or standard roster.
#11) Compares with actual results
#
###############################################################################

###########################
# 0) Install/Load Packages
###########################
options(repos = c(CRAN = "https://cloud.r-project.org"))
update.packages(ask = FALSE)
# options(warn = -1)

packages_needed <- c(
  "dplyr",
  "tidyr",
  "ggplot2",
  "glmnet",
  "randomForest",
  "xgboost",
  "Matrix",
  "nflreadr",
  "purrr",
  "ranger",
  "caret",
  "doParallel"
  # "tidymodels" # alternative to caret
)

for(p in packages_needed){
  if(!requireNamespace(p, quietly=TRUE))
    install.packages(p)
}

library(dplyr)
library(tidyr)
library(ggplot2)
library(glmnet)
library(randomForest)
library(xgboost)
library(Matrix)
library(nflreadr)
library(purrr)
library(caret)
library(doParallel)
num_cores <- parallel::detectCores()
cl <- makeCluster(num_cores - 1)
registerDoParallel(cl)


###########################
# 1) LOAD + MERGE DATA
###########################
cat("\n=== 1) LOADING MAIN NFL OFFENSE DATA (2010-2022) ===\n")
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

###########################
# 1a) Load External Data (Placeholders)
###########################
cat("\n=== Loading External Data: injuries, coaching, O-line ranks, usage stats, etc. ===\n")

# We'll pretend we have CSVs like "injuries.csv", "coaching.csv", "oline_ranks.csv", etc.
# Each has (player_id, season, or team_id, plus relevant columns).
# You must adapt to your real data structure.

# For demonstration, we’ll just create dummy placeholders:

# df_injuries <- read.csv("injuries.csv") %>% ...
# df_coaching <- read.csv("coaching.csv") %>% ...
# df_oline    <- read.csv("oline_ranks.csv") %>% ...
# df_usage    <- read.csv("usage.csv") %>% ...
# etc.

# Instead, we'll do something like:
df_injuries <- df_predict %>%
  select(player_id, season) %>%
  mutate(injury_flag = sample(c(0,1), n(), replace=TRUE, prob=c(0.8, 0.2)),
         games_missed= ifelse(injury_flag==1, sample(1:8, n(), replace=TRUE), 0))

df_oline <- df_predict %>%
  select(player_id, season) %>%
  mutate(team_oline_rank = sample(1:32, n(), replace=TRUE))

df_coaching <- df_predict %>%
  select(player_id, season) %>%
  mutate(new_head_coach = sample(c(0,1), n(), replace=TRUE, prob=c(0.9,0.1)))

# etc. For demonstration only.

cat("\n=== Merging External Data into df_predict ===\n")
df_predict <- df_predict %>%
  left_join(df_injuries, by=c("player_id","season")) %>%
  left_join(df_oline,    by=c("player_id","season"), suffix=c("", "_oline")) %>%
  left_join(df_coaching, by=c("player_id","season"), suffix=c("", "_coach"))

# Now df_predict has columns: injury_flag, games_missed, team_oline_rank, new_head_coach

###########################
# 2) ADVANCED FEATURE ENGINEERING
###########################
cat("\n=== Creating advanced features ===\n")
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

    # Incorporate the external data we merged
    was_injured        = ifelse(injury_flag==1, 1, 0),
    missed_games_frac  = games_missed / 16,  # fraction of season missed
    # For QBs, maybe penalize missing time
    adj_pass_yds       = ifelse(position=="QB", passing_yards*(1 - missed_games_frac), passing_yards),

    # Use O-line rank for QBs/RBs
    # lower rank is better (1=best), so invert it or scale:
    oline_score        = ifelse(position %in% c("QB","RB"),
                                1 / (1+team_oline_rank),
                                0.5), # default or something for WR/TE
    # Coaching change indicator
    new_coach_factor   = new_head_coach,

    # Weighted multi-year average (dummy example):
    # We'll pretend "ppr_this_season" from prior season is in the data
    # You can do real multi-year merges if you want
    # We'll skip that for brevity, but you get the idea
  )

# Now df_predict is quite large and has many advanced columns.

###########################
# 3) SPLIT BY POSITION
###########################
df_qb <- df_predict %>% filter(position=="QB")
df_rb <- df_predict %>% filter(position=="RB")
df_wr <- df_predict %>% filter(position=="WR")
df_te <- df_predict %>% filter(position=="TE")

###########################
# 3a) BUILD POSITION-SPECIFIC FORMULAS
###########################
cat("\n=== Defining advanced formulas for each position ===\n")

formula_qb <- as.formula("
  ppr_next_season ~ 
    passing_yards + passing_tds + interceptions +
    completions + pass_attempts + fumbles_total + 
    passing_efficiency + completion_rate + adj_pass_yds +
    oline_score + new_coach_factor + missed_games_frac
")

formula_rb <- as.formula("
  ppr_next_season ~ 
    carries + rushing_yards + rushing_tds + 
    receptions + receiving_yards + receiving_tds +
    fumbles_total + rushing_efficiency + over_10rush_game +
    oline_score + new_coach_factor + missed_games_frac
")

formula_wr <- as.formula("
  ppr_next_season ~
    receptions + receiving_yards + receiving_tds +
    rushing_yards + rushing_tds + fumbles_total +
    catch_efficiency + new_coach_factor + missed_games_frac
    # could add qb quality, team pass attempts, etc.
")

formula_te <- as.formula("
  ppr_next_season ~
    receptions + receiving_yards + receiving_tds +
    fumbles_total + catch_efficiency + missed_games_frac + new_coach_factor
")

###########################
# 3b) Helper to get X matrix
###########################
get_matrix <- function(df, formula_obj){
  X <- model.matrix(formula_obj, data=df)
  X <- X[, -1, drop=FALSE]  # drop intercept
  y <- df$ppr_next_season
  list(x=X, y=y)
}

###########################
# 4) CARET FOR TUNING (Optional) or Manual
###########################
# We'll do a mixture approach: partial caretaker usage + manual xgb / rf grids if we want.
# Or we can do caretaker for everything.

###########################
# 4) CREATE MASTER "fit_and_eval" that does Lasso/Ridge/Elastic + RF + XGB
#    with advanced grid search
###########################

fit_and_eval_position <- function(train_df, test_df, formula_obj){
  
  train_list <- get_matrix(train_df, formula_obj)
  test_list  <- get_matrix(test_df, formula_obj)
  
  x_train <- train_list$x
  y_train <- train_list$y
  x_test  <- test_list$x
  y_test  <- test_list$y
  
  out <- list()
  
  ##################
  # (a) Lasso, Ridge, Enet
  ##################
  # We'll do a caretaker approach or manual approach.

  # Let's do a caretaker approach for Lasso (alpha=1):
  # create caretaker trainControl
  trCtrl <- trainControl(method="cv", number=5, allowParallel=TRUE)
  
  # Lasso
  set.seed(101)
  lasso_tune <- expand.grid(alpha=1, lambda=seq(0.001,1,length=10))
  lasso_fit <- caret::train(
    x=x_train,
    y=y_train,
    method="glmnet",
    trControl=trCtrl,
    tuneGrid=lasso_tune
  )
  pred_lasso <- predict(lasso_fit, newdata=x_test)
  mse_lasso <- mean((y_test - pred_lasso)^2)
  out$mse_lasso <- mse_lasso
  out$pred_lasso <- as.vector(pred_lasso)
  
  # Ridge
  set.seed(102)
  ridge_tune <- expand.grid(alpha=0, lambda=seq(0.001,1,length=10))
  ridge_fit <- caret::train(
    x=x_train,
    y=y_train,
    method="glmnet",
    trControl=trCtrl,
    tuneGrid=ridge_tune
  )
  pred_ridge <- predict(ridge_fit, newdata=x_test)
  mse_ridge <- mean((y_test - pred_ridge)^2)
  out$mse_ridge <- mse_ridge
  out$pred_ridge <- as.vector(pred_ridge)
  
  # Elastic Net: We'll do alpha in (0.1..0.9)
  set.seed(103)
  en_tune <- expand.grid(alpha=seq(0.1,0.9,by=0.2), lambda=seq(0.001,1,length=10))
  en_fit <- caret::train(
    x=x_train,
    y=y_train,
    method="glmnet",
    trControl=trCtrl,
    tuneGrid=en_tune
  )
  pred_en <- predict(en_fit, newdata=x_test)
  mse_en <- mean((y_test - pred_en)^2)
  out$mse_elastic <- mse_en
  out$pred_elastic <- as.vector(pred_en)

  # (b) Random Forest
    set.seed(104)

    # 1) Figure out how many features we have
    n_features <- ncol(x_train)   # number of columns in x_train

    # 2) Create an mtry grid that never exceeds n_features
    mtry_candidates <- c(2,4,6,8)
    # mtry_candidates <- mtry_candidates[mtry_candidates <= n_features]  # drop large ones

    rf_grid <- expand.grid(
    mtry = mtry_candidates,
    splitrule = c("variance"),
    min.node.size = c(5,10)
    )

    rf_fit <- caret::train(
    x = x_train,
    y = y_train,
    method = "ranger",
    trControl = trCtrl,
    tuneGrid = rf_grid,
    num.trees = 500
    )

    pred_rf <- predict(rf_fit, newdata=x_test)
    mse_rf  <- mean((y_test - pred_rf)^2)
    out$mse_rf <- mse_rf
    out$pred_rf <- as.vector(pred_rf)

  # (c) XGBoost
  # caretaker method="xgbTree" is common
  set.seed(105)
  xgb_grid <- expand.grid(
    nrounds = c(200,400),
    max_depth = c(3,5,7),
    eta = c(0.02,0.1),
    gamma = c(0,1),
    colsample_bytree = c(0.8,1),
    min_child_weight = c(1,3),
    subsample = c(0.8,1)
  )
  suppressWarnings(xgb_fit <- caret::train(
    x=x_train,
    y=y_train,
    method="xgbTree",
    trControl=trCtrl,
    tuneGrid=xgb_grid,
    verbose=FALSE
  ))
  pred_xgb <- predict(xgb_fit, newdata=x_test)
  mse_xgb <- mean((y_test - pred_xgb)^2)
  out$mse_xgb <- mse_xgb
  out$pred_xgb <- as.vector(pred_xgb)
  
  # (d) LM (plain linear)
  # We'll just do a quick:
  df_lm_train <- data.frame(y=y_train, x_train)
  lm_fit <- lm(y ~ ., data=df_lm_train)
  df_lm_test <- data.frame(x_test)
  pred_lm <- predict(lm_fit, newdata=df_lm_test)
  mse_lm <- mean((y_test - pred_lm)^2)
  out$mse_lm <- mse_lm
  out$pred_lm <- pred_lm
  
  out$best_model <- NA  # we'll fill after we pick
  return(out)
}

###########################
# 5) CROSS-VALIDATION by YEAR, PER POSITION
###########################
cat("\n=== CROSS-VALIDATION by year, for each position ===\n")
all_seasons <- sort(unique(df_predict$season))
valid_seasons <- all_seasons[all_seasons <= 2021]

cv_results_all <- list(QB=data.frame(), RB=data.frame(), WR=data.frame(), TE=data.frame())

position_list <- c("QB","RB","WR","TE")
get_formula_for_pos <- function(pos){
  if(pos=="QB") return(formula_qb)
  if(pos=="RB") return(formula_rb)
  if(pos=="WR") return(formula_wr)
  if(pos=="TE") return(formula_te)
  stop("Unknown pos in get_formula_for_pos")
}

for(pos in position_list){
  cat("\n=== CV for Position:", pos, "===\n")
  df_pos <- df_predict %>% filter(position==pos)
  pos_cv_df <- data.frame(
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
  formula_pos <- get_formula_for_pos(pos)
  
  for(ty in valid_seasons){
    train_df <- df_pos %>% filter(season < ty)
    test_df  <- df_pos %>% filter(season == ty)
    if(nrow(test_df)==0 || nrow(train_df)==0) next
    
    cat(" -> POS:", pos, ", test_year:", ty,
        ", train_size:", nrow(train_df), " test_size:", nrow(test_df), "\n")
    
    out_pos <- fit_and_eval_position(train_df, test_df, formula_pos)
    
    # pick best
    model_names <- c("lm","lasso","ridge","elastic","rf","xgb")
    mse_vals    <- c(out_pos$mse_lm, out_pos$mse_lasso, out_pos$mse_ridge,
                     out_pos$mse_elastic, out_pos$mse_rf, out_pos$mse_xgb)
    best_idx <- which.min(mse_vals)
    best_mod <- model_names[best_idx]
    
    pos_cv_df <- rbind(pos_cv_df, data.frame(
      test_year   = ty,
      mse_lm      = out_pos$mse_lm,
      mse_lasso   = out_pos$mse_lasso,
      mse_ridge   = out_pos$mse_ridge,
      mse_elastic = out_pos$mse_elastic,
      mse_rf      = out_pos$mse_rf,
      mse_xgb     = out_pos$mse_xgb,
      best_model  = best_mod,
      stringsAsFactors=FALSE
    ))
  }
  cv_results_all[[pos]] <- pos_cv_df
}

###########################
# Print CV Summaries
###########################
for(pos in position_list){
  cat("\n--- CV RESULTS for", pos, "---\n")
  print(cv_results_all[[pos]])
}

###########################
# 6) FINAL TRAIN on [2010..2020], TEST on 2021
###########################
cat("\n=== FINAL TRAIN [2010..2020], TEST on 2021 (Position by Position) ===\n")
train_final <- df_predict %>% filter(season < 2021)
test_final  <- df_predict %>% filter(season == 2021)

final_pred_list <- list()

for(pos in position_list){
  df_train_pos <- train_final %>% filter(position==pos)
  df_test_pos  <- test_final %>% filter(position==pos)
  
  if(nrow(df_test_pos)==0 || nrow(df_train_pos)==0) next
  
  formula_pos <- get_formula_for_pos(pos)
  cat("\n--- Position:", pos, "---\n")
  out_final <- fit_and_eval_position(df_train_pos, df_test_pos, formula_pos)
  
  # pick best
  model_names <- c("lm","lasso","ridge","elastic","rf","xgb")
  mse_vals    <- c(out_final$mse_lm, out_final$mse_lasso, out_final$mse_ridge,
                   out_final$mse_elastic, out_final$mse_rf, out_final$mse_xgb)
  best_idx <- which.min(mse_vals)
  best_mod <- model_names[best_idx]
  cat("BEST MODEL for", pos, "with MSE =", round(min(mse_vals),2), ":", best_mod, "\n")
  
  # We might also do a "top 2 or 3" and ensemble them. Let's do that:
  sorted_idx <- order(mse_vals)
  top2_idx <- sorted_idx[1:2]
  cat("Top 2 models for", pos, "are", model_names[top2_idx[1]], "and", model_names[top2_idx[2]], "\n")
  
  # get predictions from top 2
  preds_list <- list(
    lm      = out_final$pred_lm,
    lasso   = out_final$pred_lasso,
    ridge   = out_final$pred_ridge,
    elastic = out_final$pred_elastic,
    rf      = out_final$pred_rf,
    xgb     = out_final$pred_xgb
  )
  pred_top1 <- preds_list[[model_names[top2_idx[1]]]]
  pred_top2 <- preds_list[[model_names[top2_idx[2]]]]
  
  # ensemble: average
  pred_ensemble <- 0.5*pred_top1 + 0.5*pred_top2
  
  # We'll store the ensemble as "final predicted ppr"
  # or you can just store the best one, but let's do ensemble:
  df_test_pos$pred_ppr_position <- pred_ensemble
  
  final_pred_list[[pos]] <- df_test_pos
}

###########################
# 7) Combine All Positions
###########################
cat("\n=== COMBINING ALL POSITIONS ===\n")
df_final_pred <- do.call(rbind, final_pred_list)

df_final_pred <- df_final_pred %>%
  mutate(
    pred_rank_all     = min_rank(desc(pred_ppr_position)),
    actual_rank_all   = min_rank(desc(ppr_next_season))
  )

###########################
# 8) Evaluate Top 10 ignoring positions
###########################
cat("\n=== MODEL's Top 10 (Any Position) ===\n")
top10_model <- df_final_pred %>%
  filter(pred_rank_all <= 10) %>%
  arrange(pred_rank_all)

print(top10_model %>%
  select(pred_rank_all, player_display_name, position, ppr_next_season, pred_ppr_position))

cat("\n=== ACTUAL Top 10 (Any Position) ===\n")
top10_actual <- df_final_pred %>%
  filter(actual_rank_all <= 10) %>%
  arrange(actual_rank_all)

print(top10_actual %>%
  select(actual_rank_all, player_display_name, position, ppr_next_season, pred_ppr_position))

overlap_ids <- intersect(top10_model$player_id, top10_actual$player_id)
cat("\nOverlap in top-10:", length(overlap_ids), "players\n")

sum_model <- sum(top10_model$ppr_next_season, na.rm=TRUE)
sum_actual <- sum(top10_actual$ppr_next_season, na.rm=TRUE)
cat("\nPoints by predicted top-10:", round(sum_model,1),
    "\nPoints by actual top-10:   ", round(sum_actual,1),
    "\nDifference:                ", round(sum_actual - sum_model,1), "\n")

###########################
# 9) Evaluate Standard Roster
###########################
cat("\n=== STANDARD ROSTER (1 QB, 2 RB, 2 WR, 1 TE) ===\n")
roster_needs <- c(QB=1, RB=2, WR=2, TE=1)

team_picks <- list()
for(pos in names(roster_needs)){
  needed <- roster_needs[pos]
  subdf <- df_final_pred %>%
    filter(position==pos) %>%
    arrange(desc(pred_ppr_position))
  picks <- head(subdf, needed)
  team_picks[[pos]] <- picks
}
team_df <- do.call(rbind, team_picks)
team_score <- sum(team_df$ppr_next_season, na.rm=TRUE)
cat("Our model-based roster actual PPR:", round(team_score,1), "\n")

# Compare to best actual
best_team_picks <- list()
for(pos in names(roster_needs)){
  needed <- roster_needs[pos]
  subdf <- df_final_pred %>%
    filter(position==pos) %>%
    arrange(desc(ppr_next_season))
  picks <- head(subdf, needed)
  best_team_picks[[pos]] <- picks
}
best_team_df <- do.call(rbind, best_team_picks)
best_team_score <- sum(best_team_df$ppr_next_season, na.rm=TRUE)

cat("Best possible actual roster (same constraints):", round(best_team_score,1), "\n")
cat("Difference:", round(best_team_score - team_score,1), "\n")

###########################
# 10) DONE
###########################
cat("\n*** COMPLETE 'KITCHEN SINK' PIPELINE FINISHED ***\n")
stopCluster(cl)