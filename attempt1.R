#!/usr/bin/env Rscript
# Initial attempt without prediction. Just caried over some stuff from problem set code.

options(repos = c(CRAN = "https://cloud.r-project.org"))

packages_needed <- c("nflreadr",
                     "dplyr",
                     "ggplot2",
                     "tidyr",
                     "gamlr",
                     "glmnet",
                     "Matrix",
                     "broom",
                     "factoextra",
                     "randomForest",
                     "car")

for (pkg in packages_needed) {
  if (!requireNamespace(pkg, quietly=TRUE))
    install.packages(pkg)
}

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
library(car)

seasons_to_load <- c(2021, 2022)
df_raw <- load_player_stats(stat_type = "offense", seasons = seasons_to_load)
cat("=== HEAD OF COMBINED DATAFRAME ===\n")
print(head(df_raw))
cat("\n=== STRUCTURE OF DATAFRAME ===\n")
str(df_raw)
cat("\n=== SUMMARY STATS OF DATAFRAME ===\n")
print(summary(df_raw))

df <- df_raw %>%
  filter(position %in% c("QB","RB","WR","TE")) %>%
  mutate(
    total_touches = carries + receptions,
    total_opps    = carries + targets
  ) %>%
  filter(total_opps >= 10)
cat("\n=== AFTER FILTERING FOR QB/RB/WR/TE & >= 10 opportunities ===\n")
print(summary(df))

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

cat("\n=== SUMMARY STATS: Player-Season Dataset ===\n")
print(summary(df_season))
cat("\n=== TOP 10 PLAYERS BY RECEPTIONS ===\n")
df_season %>%
  arrange(desc(receptions)) %>%
  slice_head(n=10) %>%
  print()

ggplot(df_season, aes(x=fantasy_points_ppr)) +
  geom_histogram(bins=30, alpha=0.7) +
  labs(title="Distribution of PPR Fantasy Points", x="PPR Points", y="Count")

ggplot(df_season, aes(x=receptions, y=fantasy_points_ppr, color=position)) +
  geom_point(alpha=0.6) +
  theme_minimal() +
  labs(title="Receptions vs. PPR Points by Position")

lm_simple <- lm(
  fantasy_points_ppr ~ rushing_yards + receiving_yards + receptions + passing_yards,
  data = df_season
)
cat("\n=== Multiple Linear Regression (Simple) Summary ===\n")
print(summary(lm_simple))

lm_full <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = df_season
)
cat("\n=== Multiple Linear Regression (Full) Summary ===\n")
print(summary(lm_full))

cat("\n=== Compare AIC and BIC for Simple vs. Full Models ===\n")
cat("AIC(simple):", AIC(lm_simple), "\n")
cat("AIC(full)  :", AIC(lm_full), "\n")
cat("BIC(simple):", BIC(lm_simple), "\n")
cat("BIC(full)  :", BIC(lm_full), "\n")

cat("\n=== VIF for Full Model ===\n")
print(vif(lm_full))

step_forward <- step(lm_simple,
                     scope = formula(lm_full),
                     direction = "forward",
                     trace = FALSE)
cat("\n=== Forward Stepwise Selected Model ===\n")
print(summary(step_forward))

set.seed(1001)
n <- nrow(df_season)
train_idx <- sample(seq_len(n), size = floor(0.7*n))
train_data <- df_season[train_idx, ]
test_data  <- df_season[-train_idx, ]

lm_oos <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = train_data
)
test_pred <- predict(lm_oos, newdata=test_data)
R2 <- function(pred, obs){
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}
oos_r2 <- R2(test_pred, test_data$fantasy_points_ppr)
cat("\n=== OOS R^2 for Full LM on Test Set ===\n", oos_r2, "\n")

model_vars <- c("completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
                "receiving_yards", "receiving_tds")
trainX <- as.matrix(train_data[, model_vars])
trainY <- train_data$fantasy_points_ppr
testX  <- as.matrix(test_data[, model_vars])
testY  <- test_data$fantasy_points_ppr

set.seed(2023)
cv_lasso <- cv.glmnet(trainX, trainY, alpha=1)
best_lambda_lasso <- cv_lasso$lambda.min
cat("\n=== Best lambda (LASSO) from cross-validation ===\n", best_lambda_lasso, "\n")
coef_lasso <- coef(cv_lasso, s="lambda.min")
cat("\n=== Coefficients (LASSO) at lambda.min ===\n")
print(coef_lasso)
pred_lasso <- predict(cv_lasso, newx=testX, s="lambda.min")
r2_lasso   <- R2(pred_lasso, testY)
cat("\n=== LASSO OOS R^2 ===\n", r2_lasso, "\n")

set.seed(2024)
cv_ridge <- cv.glmnet(trainX, trainY, alpha=0)
best_lambda_ridge <- cv_ridge$lambda.min
cat("\n=== Best lambda (Ridge) from cross-validation ===\n", best_lambda_ridge, "\n")
coef_ridge <- coef(cv_ridge, s="lambda.min")
cat("\n=== Coefficients (Ridge) at lambda.min ===\n")
print(coef_ridge)
pred_ridge <- predict(cv_ridge, newx=testX, s="lambda.min")
r2_ridge   <- R2(pred_ridge, testY)
cat("\n=== Ridge OOS R^2 ===\n", r2_ridge, "\n")

df_cluster <- df_season %>%
  select(player_id, player_display_name, position,
         rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fantasy_points_ppr)
df_kmeans_input <- df_cluster %>%
  select(rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds, fantasy_points_ppr)
df_scaled <- scale(df_kmeans_input)

set.seed(999)
km3 <- kmeans(df_scaled, centers=3, nstart=25)
df_clustered <- df_cluster %>%
  mutate(cluster = factor(km3$cluster))
cat("\n=== Distribution of Clusters ===\n")
print(table(df_clustered$cluster))
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

df_season <- df_season %>%
  mutate(
    rank_ppr = rank(-fantasy_points_ppr, ties.method="first"),
    isTop24   = ifelse(rank_ppr <= 24, 1, 0)
  )

set.seed(123)
n2 <- nrow(df_season)
train_idx2 <- sample(seq_len(n2), size = floor(0.7 * n2))
train2 <- df_season[train_idx2, ]
test2  <- df_season[-train_idx2, ]

logit_mod <- glm(
  isTop24 ~ completions + attempts + passing_yards + passing_tds +
            carries + rushing_yards + rushing_tds +
            receptions + receiving_yards + receiving_tds,
  data=train2,
  family=binomial
)
cat("\n=== Logistic Model (Top 24 vs. Others) ===\n")
print(summary(logit_mod))
prob_test <- predict(logit_mod, newdata=test2, type="response")
pred_class <- ifelse(prob_test > 0.5, 1, 0)
accuracy <- mean(pred_class == test2$isTop24)
cat("\n=== Classification Accuracy (Top 24) ===\n")
cat("Accuracy:", round(accuracy, 3), "\n")

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
rf_pred <- predict(rf_model, newdata=test_data)
r2_rf   <- R2(rf_pred, test_data$fantasy_points_ppr)
cat("\n=== RF OOS R^2 ===\n", r2_rf, "\n")
cat("\n=== Random Forest Variable Importance ===\n")
print(importance(rf_model))
varImpPlot(rf_model, main="Random Forest Variable Importance")

lm_for_fdr <- lm(
  fantasy_points_ppr ~ completions + attempts + passing_yards + passing_tds + interceptions +
    carries + rushing_yards + rushing_tds + targets + receptions + receiving_yards + receiving_tds,
  data = df_season
)
lm_summary <- summary(lm_for_fdr)
pvals <- lm_summary$coefficients[-1, 4]

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
