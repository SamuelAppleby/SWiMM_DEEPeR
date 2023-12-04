library(data.table)
library(reshape2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(scipen = 0)
options(digits = 3)

df_resize <- read.csv(file = "C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/data/image_similarity/results/resizing.csv")
df_resize <- subset(df_resize, select=c(width, height, cosine_similarity))

df_resize$resolution <- ifelse(df_resize$width == 1920, "high_raw_sim", "low_raw_sim")
df_resize <- subset(df_resize, select=c(resolution, cosine_similarity))

calculate_summary_stats <- function(res) {
  subset_df <- subset(df_resize, resolution == res)
  c(
    mean = mean(subset_df$cosine_similarity),
    sd = sd(subset_df$cosine_similarity),
    gamma = 1 - mean(subset_df$cosine_similarity)
  )
}

df_resize <- as.data.frame(sapply(c("high_raw_sim", "low_raw_sim"), calculate_summary_stats))

df_cross_res <- read.csv(file = "C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/data/image_similarity/results/cross_resolution.csv")

df_cross_res <- subset(df_cross_res, select=c(low_high_sim, low_raw_sim, high_raw_sim))

df_cross_res <- sapply(df_cross_res, function(x) c(
  "mean" = mean(x),
  "sd" = sd(x),
  "gamma" = 1 - mean(x)
)
)

df_cross_res <- as.data.frame(df_cross_res)

# df_inference <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/logs/sac/1/inference.csv')
# 
# df_inference <- sapply(df_inference, function(x) c(
#   "mean" = mean(x),
#   "sd" = sd(x),
#   "n" = length(x)
# )
# )