library(utils)
library(data.table)
library(ggplot2)
library(yaml)
library(scales)
library(extrafont)
library(tidyr)

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

directory_path <- list(
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\sac\\sac_1\\final_model_metrics",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\ppo\\ppo_3\\final_model_metrics",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\td3\\td3_4\\final_model_metrics"
)

combined_metric_data <- data.frame()

for (algo_dir in directory_path) {
  folders <- list.dirs(algo_dir, full.names = TRUE, recursive = FALSE)
  
  for (seed_dir in folders) {
    metric_data <- read.csv(file.path(seed_dir, "final_model_metrics.csv"))
    combined_metric_data <- rbind(combined_metric_data,metric_data)
  }
}

combined_metric_data$Episode <- NULL
combined_metric_data$Seed <- NULL

# Sum for specific columns
sum_data <- aggregate(cbind(OutOfView, MaximumDistance, TargetCollision) ~ Algorithm, 
                      data = combined_metric_data, FUN = sum, na.rm = TRUE)

df_long <- sum_data %>%
  pivot_longer(cols = c(OutOfView, MaximumDistance, TargetCollision), names_to = "Variable", values_to = "Value")

df_long$Algorithm <- factor(df_long$Algorithm, levels = c("SAC", "PPO", "TD3"))
df_long$Variable <- factor(df_long$Variable, levels = c("OutOfView", "MaximumDistance", "TargetCollision"))

ggplot(df_long, aes(x = Algorithm, y = Value, fill = Variable)) +
  scale_y_continuous(name = "Count") +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(x = "Algorithm", y = "Count", fill = "Fatal Metric") +
  scale_fill_brewer(palette = "Set2", name = "Variable", labels = c("OutOfView" = "Out Of View",
                                                                    "MaximumDistance" = "Maximum Distance",
                                                                    "TargetCollision" = "Target Collision")) +
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"))

# Mean for all other columns
mean_data <- aggregate(. ~ Algorithm,
                       data = combined_metric_data[, !(names(combined_metric_data) %in% c("OutOfView", "MaximumDistance", "TargetCollision"))],
                       FUN = mean, na.rm = TRUE)

# Merge the results
combined_metric_data_summary <- merge(sum_data, mean_data, by = c("Algorithm"))

