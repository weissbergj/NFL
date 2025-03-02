#!/usr/bin/env Rscript

# 1) Specify a CRAN mirror for non-interactive installs
options(repos = c(CRAN = "https://cloud.r-project.org"))

# 2) Install needed packages (comment these out after the first successful run)
install.packages("nflreadr")
install.packages("dplyr")

# 3) Load the libraries
library(nflreadr)
library(dplyr)

# 4) Load 2022 Offensive Player Stats (weekly by default in nflreadr v1.4.x+)
df_weekly <- load_player_stats(
  stat_type = "offense",
  seasons   = 2022
)

# 5) Inspect the Data
cat("=== HEAD OF DF_WEEKLY ===\n")
head(df_weekly)

cat("\n=== STRUCTURE OF DF_WEEKLY ===\n")
str(df_weekly)

cat("\n=== SAMPLE SUMMARY STATS ===\n")
summary(df_weekly)

# 6) Top 10 Players by Total Receptions
cat("\n=== TOP 10 PLAYERS BY TOTAL RECEPTIONS ===\n")
df_weekly %>%
  group_by(player_display_name) %>%
  summarize(total_rec = sum(receptions, na.rm = TRUE)) %>%
  arrange(desc(total_rec)) %>%
  slice_head(n = 10) %>%
  print()

# 7) Summaries by Position
cat("\n=== SUMMARY BY POSITION ===\n")
df_weekly %>%
  group_by(position) %>%
  summarize(
    total_receptions    = sum(receptions, na.rm = TRUE),
    total_rushing_yds   = sum(rushing_yards, na.rm = TRUE),
    total_receiving_yds = sum(receiving_yards, na.rm = TRUE),
    total_touchdowns    = sum(rushing_tds + receiving_tds + passing_tds, na.rm = TRUE)
  ) %>%
  arrange(desc(total_receptions)) %>%
  print()
