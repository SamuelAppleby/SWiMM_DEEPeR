library(data.table)
library(reshape2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_resize <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/benchmarking/results/resizing_low.csv')

df_resize <- subset(df_resize, select=c(width, height, cosine_similarity))

df_resize <- aggregate(. ~ width + height, df_resize, function(x) c(
  mean = mean(x), 
  sd = sd(x)
  )
  )


# df_cross_res <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/benchmarking/results/cross_resolution.csv')
# 
# df_cross_res <- subset(df_cross_res, select=c(low_high_sim, low_raw_sim, high_raw_sim))
# 
# df_cross_res <- sapply(df_cross_res, function(x) c(
#   "Mean" = mean(x),
#   "Stand dev" = sd(x),
#                          "n" = length(x)
# 
# )
# )


df_inference <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/Logs/sac/1/inference.csv')

df_inference <- sapply(df_inference, function(x) c(
  "mean" = mean(x),
  "sd" = sd(x),
  "n" = length(x)
)
)