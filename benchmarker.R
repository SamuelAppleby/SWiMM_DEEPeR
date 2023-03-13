library(data.table)
library(reshape2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_resize <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/benchmarking_results/resizing.csv')

df_resize <- subset(df_resize, select=c(width, height, cosine_similarity))

df_resize <- aggregate(. ~ width + height, df_resize, function(x) c(mean = mean(x), sd = sd(x)))



df_cross_res <- read.csv(file = 'C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR/benchmarking_results/cross_resolution.csv')

df_cross_res <- subset(df_cross_res, select=c(low_high_sim, low_raw_sim, high_raw_sim))


df_cross_res <- sapply(df_cross_res, function(x) c( 
  "Mean" = mean(x),
  "Stand dev" = sd(x), 
                         "n" = length(x)

)
)