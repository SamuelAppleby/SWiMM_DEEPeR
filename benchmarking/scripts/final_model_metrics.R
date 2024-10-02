library(utils)
library(data.table)
library(ggplot2)
library(scales)
library(extrafont)
library(tidyr)

windowsFonts(
  CAL=windowsFont("Parisienne")
)

scientific_10 <- function(x, y) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x / (10^y))))
}

# Wrapper function to allow y to be specified
scientific_10_wrapper <- function(y) {
  function(x) scientific_10(x, y)
}

directory_path <- list(
  "C:\\Users\\sambu\\Downloads\\SWiMM_DEEPeR_old\\data",
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

agg_step <- aggregate(cbind(OutOfView, MaximumDistance, TargetCollision) ~ Algorithm + Seed + Episode, 
                      data = combined_metric_data, FUN = max, na.rm = TRUE)

# Sum for specific columns
sum_data <- aggregate(cbind(OutOfView, MaximumDistance, TargetCollision) ~ Algorithm, 
                      data = agg_step, FUN = sum, na.rm = TRUE)

df_long <- sum_data %>%
  pivot_longer(cols = c(OutOfView, MaximumDistance, TargetCollision), names_to = "Variable", values_to = "Value")

df_long$Algorithm <- factor(df_long$Algorithm, levels = c("SAC_OLD", "SAC", "PPO", "TD3"))
df_long$Variable <- factor(df_long$Variable, levels = c("OutOfView", "MaximumDistance", "TargetCollision"))

ggplot(df_long, aes(x = Algorithm, y = Value, fill = Variable)) +
  scale_x_discrete(labels = c("SAC_OLD" = expression(SAC[paste(1.0)]),
                              "SAC" = expression(SAC[paste(2.0)])))+
  scale_y_continuous(name = "Occurences") +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(x = "Algorithm", y = "Occurences", fill = "Fatal Metric") +
  scale_fill_brewer(palette = "Set2", name = "Fatal Criterion", labels = c("OutOfView" = "A'",
                                                                    "MaximumDistance" = "D'",
                                                                    "TargetCollision" = "C'")) +
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"),
        legend.text = element_text(family = "CAL"))

# Mean for all other columns
mean_data <- aggregate(. ~ Algorithm,
                       data = combined_metric_data[, !(names(combined_metric_data) %in% c("Seed", "Episode", "Step", "OutOfView", "MaximumDistance", "TargetCollision"))],
                       FUN = mean, na.rm = TRUE)

# Merge the results
combined_metric_data_summary <- merge(sum_data, mean_data, by = c("Algorithm"))

df_long_2 <- combined_metric_data_summary %>%
  pivot_longer(cols = c(AError, DError, ASmoothnessError, DSmoothnessError), names_to = "Variable", values_to = "Value")

df_long_2$Algorithm <- factor(df_long_2$Algorithm, levels = c("SAC_OLD", "SAC", "PPO", "TD3"))

increase <- 1

ggplot(subset(df_long_2, Variable %in% c("AError", "DError")), aes(x = Algorithm, y = Value * (10 ^ increase), fill = Variable)) +
  scale_x_discrete(labels = c("SAC_OLD" = expression(SAC[paste(1.0)]),
                              "SAC" = expression(SAC[paste(2.0)])))+
  scale_y_log10(labels = scientific_10_wrapper(y = increase)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(x = "Algorithm", y = "MAE") +
  scale_fill_brewer(palette = "Set2", name = "Error Type", labels = c("AError" = "A",
                                                                           "DError" = "D")) +
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"),
        legend.text = element_text(family = "CAL"))

increase <- 3

ggplot(subset(df_long_2, Variable %in% c("ASmoothnessError", "DSmoothnessError")), aes(x = Algorithm, y = Value * (10 ^ increase), fill = Variable)) +
  scale_x_discrete(labels = c("SAC_OLD" = expression(SAC[paste(1.0)]),
                              "SAC" = expression(SAC[paste(2.0)])))+
  scale_y_log10(labels = scientific_10_wrapper(y = increase)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(x = "Algorithm", y = "MAE") +
  scale_fill_brewer(palette = "Set2", name = "Smoothness Error Type", labels = c("ASmoothnessError" = expression(S[A]),
                                                                  "DSmoothnessError" = expression(S[D]))) +
  theme(legend.position = "bottom",
        text = element_text(family = "Times New Roman"),
        legend.text = element_text(family = "CAL"))
