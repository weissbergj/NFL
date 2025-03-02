#!/usr/bin/env Rscript
###############################################################################
# ECON 108 Final Project
# 
# "Waking the Sleepers: Using Data Science to Discover Undervalued Players 
#  in PPR Scored Fantasy Football"
#
# Authors: Austin Bennett, Jared Weissberg
#
# COMPLETE SCRIPT: from data loading, EDA, multiple regression, 
# LASSO/Ridge, classification (K-means), and optional Random Forest
# plus an illustration of FDR-based selection of sleepers.
###############################################################################

############################
# 1) SETUP ENVIRONMENT
############################
# 1a) Choose a CRAN mirror (non-interactive)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# 1b) Install needed packages (only run if not yet installed)
#    In real usage, you may comment these out after the first successful run.
packages_needed <- c("nflreadr",   # For NFL data
                     "dplyr",      # Data manipulation
                     "ggplot2",    # Plotting
                     "tidyr",      # Data tidying
                     "gamlr",      # LASSO, Ridge with AICc built-in
                     "glmnet",     # Another option for LASSO, Ridge
                     "Matrix",     # Sparse matrices
                     "broom",      # Tidying model outputs
                     "factoextra", # For K-Means visualization
                     "randomForest", # For optional Random Forest
                     "car")        # Some helper functions (e.g. VIF)

for (pkg in packages_needed) {
  if (!requireNamespace(pkg, quietly=TRUE))
    install.packages(pkg)
}

# 1c) Load Libraries
library(nflreadr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(gamlr)
library(glmnet)
library(Matrix)
library(broom)
library(factoextra)
library(randomForest)
library(car)  # for VIF, etc.

############################
# 2) LOAD & PREP DATA
############################

# 2a) For demonstration, we'll load 2021 + 2022 Offensive Player Stats
#     If desired, you can add more seasons in the vector c(2019,2020,2021,2022).
seasons_to_load <- c(2021, 2022)
df_raw <- load_player_stats(stat_type = "offense", 
                            seasons = seasons_to_load)

cat("=== HEAD OF COMBINED DATAFRAME ===\n")
print(head(df_raw))

cat("\n=== STRUCTURE OF DATAFRAME ===\n")
str(df_raw)

cat("\n=== SUMMARY STATS OF DATAFRAME ===\n")
print(summary(df_raw))

# 2b) Filter out unusual positions if focusing only on typical fantasy positions 
#     (QB, RB, WR, TE). Also remove players with minimal touches, if desired.
df <- df_raw %>%
  filter(position %in% c("QB","RB","WR","TE")) %>%
  # Example: keep only players with >10 attempts/carries/targets, etc. 
  # (You can adjust these thresholds to avoid extremely small sample noise)
  mutate(
    total_touches = carries + receptions,
    total_opps    = carries + targets
  ) %>%
  filter(total_opps >= 10)

cat("\n=== AFTER FILTERING FOR QB/RB/WR/TE & >= 10 opportunities ===\n")
print(summary(df))

# 2c) Create a small aggregated dataset by player-season if desired
#    (some modeling can be done at the weekly level, but example here is season-level).
#    For simple demonstration, we can sum up over each player_id + season:
df_season <- df %>%
  group_by(player_id, player_display_name, position, season) %>%
  summarise(
    team                 = dplyr::last(recent_team),
    games_played         = n(),
    completions          = sum(completions, na.rm=TRUE),
    attempts             = sum(attempts, na.rm=TRUE),
    passing_yards        = sum(passing_yards, na.rm=TRUE),
    passing_tds          = sum(passing_tds, na.rm=TRUE),
    interceptions        = sum(interceptions, na.rm=TRUE),
    carries              = sum(carries, na.rm=TRUE),
    rushing_yards        = sum(rushing_yards, na.rm=TRUE),
    rushing_tds          = sum(rushing_tds, na.rm=TRUE),
    targets              = sum(targets, na.rm=TRUE),
    receptions           = sum(receptions, na.rm=TRUE),
    receiving_yards      = sum(receiving_yards, na.rm=TRUE),
    receiving_tds        = sum(receiving_tds, na.rm=TRUE),
    fantasy_points_ppr   = sum(fantasy_points_ppr, na.rm=TRUE),
    .groups = "drop"
  )

cat("\n=== HEAD OF AGGREGATED (PLAYER-SEASON) DATA ===\n")
print(head(df_season))

############################
# 3) EXPLORATORY DATA ANALYSIS (EDA)
############################

# 3a) Basic summary stats at the player-season level
cat("\n=== SUMMARY STATS: Player-Season Dataset ===\n")
print(summary(df_season))

# 3b) Check top 10 players by total receptions in aggregated dataset
cat("\n=== TOP 10 PLAYERS BY RECEPTIONS ===\n")
df_season %>%
  arrange(desc(receptions)) %>%
  slice_head(n=10) %>%
  print()

# 3c) Distribution of fantasy_points_ppr
ggplot(df_season, aes(x=fantasy_points_ppr)) +
  geom_histogram(bins=30, fill="blue", alpha=0.7) +
  labs(title="Distribution of PPR Fantasy Points", x="PPR Points", y="Count")

# 3d) Relationship of receptions vs. fantasy_points_ppr
ggplot(df_season, aes(x=receptions, y=fantasy_points_ppr, color=position)) +
  geom_point(alpha=0.6) +
  theme_minimal() +
  labs(title="Receptions vs. PPR Points by Position")

############################
# 4) MULTIPLE LINEAR REGRESSION
############################

# 4a) Build a simple linear model: fantasy_points_ppr ~ a few features
#     We exclude 'player_id' because it's an ID, but we include key stats
lm_simple <- lm(
  fantasy_points_ppr ~ rushing_yards + receiving_yards + receptions + passing_yards,
  data = df_season
)
cat("\n=== Multiple Linear Regression (Simple) Summary ===\n")
print(summary(lm_simple))

# 4b) Another more comprehensive linear model with more features
#     (You can add as many as you want, but be wary of multicollinearity.)
lm_full <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = df_season
)
cat("\n=== Multiple Linear Regression (Full) Summary ===\n")
print(summary(lm_full))

# 4c) Compare AIC, BIC of the two models
cat("\n=== Compare AIC and BIC for Simple vs. Full Models ===\n")
cat("AIC(simple):", AIC(lm_simple), "\n")
cat("AIC(full)  :", AIC(lm_full), "\n")
cat("BIC(simple):", BIC(lm_simple), "\n")
cat("BIC(full)  :", BIC(lm_full), "\n")

# 4d) Check variance inflation factors (VIF) for the full model
cat("\n=== VIF for Full Model ===\n")
print(vif(lm_full))

# 4e) (Optional) Forward or backward stepwise to see if simpler subset is suggested
#     Start from the simple model:
step_forward <- step(lm_simple, 
                     scope = formula(lm_full), 
                     direction = "forward", 
                     trace = FALSE)
cat("\n=== Forward Stepwise Selected Model ===\n")
print(summary(step_forward))

############################
# 5) TRAIN/TEST SPLIT & OOS PERFORMANCE
############################

set.seed(1001)
n <- nrow(df_season)
train_idx <- sample(seq_len(n), size = floor(0.7*n))  # 70% train
train_data <- df_season[train_idx, ]
test_data  <- df_season[-train_idx, ]

lm_oos <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = train_data
)

# Pred on test
test_pred <- predict(lm_oos, newdata=test_data)
# Out-of-sample R^2 function
R2 <- function(pred, obs){
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}
oos_r2 <- R2(test_pred, test_data$fantasy_points_ppr)
cat("\n=== OOS R^2 for Full LM on Test Set ===\n", oos_r2, "\n")

############################
# 6) REGULARIZATION: LASSO & RIDGE
############################
# We'll demonstrate with glmnet.

# 6a) Prepare matrix X, response y
#     Typically you remove the outcome from the matrix of predictors:
model_vars <- c("completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
                "receiving_yards", "receiving_tds")
trainX <- as.matrix(train_data[, model_vars])
trainY <- train_data$fantasy_points_ppr
testX  <- as.matrix(test_data[, model_vars])
testY  <- test_data$fantasy_points_ppr

# 6b) LASSO with cross-validation
set.seed(2023)
cv_lasso <- cv.glmnet(trainX, trainY, alpha=1)  # alpha=1 => LASSO
best_lambda_lasso <- cv_lasso$lambda.min
cat("\n=== Best lambda (LASSO) from cross-validation ===\n", best_lambda_lasso, "\n")

# 6c) Coefficients at best lambda
coef_lasso <- coef(cv_lasso, s="lambda.min")
cat("\n=== Coefficients (LASSO) at lambda.min ===\n")
print(coef_lasso)

# 6d) Predict on test
pred_lasso <- predict(cv_lasso, newx=testX, s="lambda.min")
r2_lasso   <- R2(pred_lasso, testY)
cat("\n=== LASSO OOS R^2 ===\n", r2_lasso, "\n")

# 6e) Ridge with cross-validation
set.seed(2024)
cv_ridge <- cv.glmnet(trainX, trainY, alpha=0)  # alpha=0 => Ridge
best_lambda_ridge <- cv_ridge$lambda.min
cat("\n=== Best lambda (Ridge) from cross-validation ===\n", best_lambda_ridge, "\n")

# 6f) Coefficients at best lambda
coef_ridge <- coef(cv_ridge, s="lambda.min")
cat("\n=== Coefficients (Ridge) at lambda.min ===\n")
print(coef_ridge)

# 6g) Predict on test
pred_ridge <- predict(cv_ridge, newx=testX, s="lambda.min")
r2_ridge   <- R2(pred_ridge, testY)
cat("\n=== Ridge OOS R^2 ===\n", r2_ridge, "\n")

############################
# 7) CLASSIFICATION EXAMPLE: K-MEANS
############################
# We will cluster players based on receiving_yards, receptions, etc. 
# (This is purely illustrative: you could define “Breakouts” vs “Top-24” more directly.)

# 7a) Choose features for clustering. For instance, receiving & rushing volume
df_cluster <- df_season %>%
  select(player_id, player_display_name, position,
         rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fantasy_points_ppr)

# 7b) We only do K-Means on numeric columns:
df_kmeans_input <- df_cluster %>%
  select(rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fantasy_points_ppr)

# 7c) Scale them (standard practice)
df_scaled <- scale(df_kmeans_input)

# 7d) Decide on number of clusters k. Let's do k=3 for demonstration 
#     (like "Stars", "Contributors", "Bench" or "Breakouts"/"Starters"/"Fliers").
set.seed(999)
km3 <- kmeans(df_scaled, centers=3, nstart=25)

# 7e) Attach cluster labels
df_clustered <- df_cluster %>%
  mutate(cluster = factor(km3$cluster))

cat("\n=== Distribution of Clusters ===\n")
print(table(df_clustered$cluster))

# 7f) Visualize two main dimensions
# fviz_cluster(km3, data=df_scaled, geom="point", main="K-Means with k=3",
            #  ellipse.type="norm")

# 7g) Quick summary of each cluster’s average stats
cluster_summary <- df_clustered %>%
  group_by(cluster) %>%
  summarise(
    count = n(),
    avg_rush_yds = mean(rushing_yards),
    avg_rush_tds = mean(rushing_tds),
    avg_rec      = mean(receptions),
    avg_rec_yds  = mean(receiving_yards),
    avg_rec_tds  = mean(receiving_tds),
    avg_ppr      = mean(fantasy_points_ppr)
  )
cat("\n=== Cluster Summary (Mean Stats) ===\n")
print(cluster_summary)

############################
# 8) BINARY CLASSIFICATION EXAMPLE (LOGISTIC)
############################

# STEP 1: Create a 'Top 24' label
# We'll define "Top 24" by total (or average) fantasy_points_ppr 
# in df_season. If you prefer a different cutoff, just change 24 -> something else.
df_season <- df_season %>%
  mutate(
    rank_ppr = rank(-fantasy_points_ppr, ties.method="first"),
    isTop24   = ifelse(rank_ppr <= 24, 1, 0)
  )

# STEP 2: Split data (70/30) for logistic classification
set.seed(123)
n2 <- nrow(df_season)
train_idx2 <- sample(seq_len(n2), size = floor(0.7 * n2))
train2 <- df_season[train_idx2, ]
test2  <- df_season[-train_idx2, ]

# STEP 3: Fit a simple logistic model
# Choose whichever predictors you like. 
# Here we mimic a structure from the aggregator:
logit_mod <- glm(
  isTop24 ~ completions + attempts + passing_yards + passing_tds + 
            carries + rushing_yards + rushing_tds + 
            receptions + receiving_yards + receiving_tds,
  data=train2,
  family=binomial
)

cat("\n=== Logistic Model (Top 24 vs. Others) ===\n")
print(summary(logit_mod))

# STEP 4: Predict on the test set and measure accuracy
prob_test <- predict(logit_mod, newdata=test2, type="response")
pred_class <- ifelse(prob_test > 0.5, 1, 0)
accuracy <- mean(pred_class == test2$isTop24)

cat("\n=== Classification Accuracy (Top 24) ===\n")
cat("Accuracy:", round(accuracy, 3), "\n")


############################
# 9) OPTIONAL: RANDOM FOREST
############################
# An ensemble method that can capture nonlinear relationships. 
# We'll treat PPR points as the outcome again.

set.seed(9999)
rf_model <- randomForest(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = train_data,
  ntree = 500,
  importance = TRUE
)
cat("\n=== Random Forest Model Summary ===\n")
print(rf_model)

# Predict on test
rf_pred <- predict(rf_model, newdata=test_data)
r2_rf   <- R2(rf_pred, test_data$fantasy_points_ppr)
cat("\n=== RF OOS R^2 ===\n", r2_rf, "\n")

# Variable Importance
cat("\n=== Random Forest Variable Importance ===\n")
print(importance(rf_model))
varImpPlot(rf_model, main="Random Forest Variable Importance")

############################
# 10) EXAMPLE: FDR ADJUSTMENTS FOR MULTIPLE TESTS
############################
# Suppose we test 20 features for significance in predicting PPR, then pick "sleepers."
# We'll do a simplistic demonstration with the FULL linear model’s summary.

lm_for_fdr <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = df_season
)
lm_summary <- summary(lm_for_fdr)
pvals <- lm_summary$coefficients[-1, 4]  # exclude intercept p-value

# A simple FDR cut function
fdr_cut <- function(pvals, q=0.1){
  pvals <- sort(pvals[!is.na(pvals)])
  N <- length(pvals)
  k <- rank(pvals, ties.method="min")
  alpha <- max(pvals[pvals <= (q * k/(N + 1))])
  alpha
}

alpha_10pct <- fdr_cut(pvals, q=0.1)
cat("\n=== FDR alpha cut for q=0.1 ===\n", alpha_10pct, "\n")

sig_features <- names(pvals)[pvals <= alpha_10pct]
cat("\n=== Features Surviving FDR @ 10% ===\n")
print(sig_features)

# If you had a hypothesis about certain players being “sleeper candidates”
# based on these significant features, you'd proceed to evaluate them 
# in a final model or compare their predicted vs. actual. 
# This is just an illustration.

############################
# 11) WRAP-UP
############################

cat("\n\n*** COMPLETE SCRIPT FINISHED ***\n")
cat("Summary of Key Results:\n")
cat(" - OOS R^2, Full LM  :", round(oos_r2, 3), "\n")
cat(" - OOS R^2, LASSO    :", round(r2_lasso, 3), "\n")
cat(" - OOS R^2, Ridge    :", round(r2_ridge, 3), "\n")
cat(" - OOS R^2, RandomForest :", round(r2_rf, 3), "\n")
cat(" - K-Means cluster summaries are in 'cluster_summary' object.\n")
cat(" - FDR discovered features (p <= ", round(alpha_10pct,4), 
    "):", paste(sig_features, collapse=", "), "\n\n")

cat("\nDEBUG: Reached the very end of the script!\n")
