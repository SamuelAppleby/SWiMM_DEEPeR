library(utils)
library(data.table)
library(ggplot2)
library(scales)
library(extrafont)
library(tidyr)
library(dplyr)

windowsFonts(
  CAL=windowsFont("Parisienne")
)

scientific_10 <- function(x, y) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x / (10^y))))
}

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
combined_testing_montior <- data.frame()

for (algo_dir in directory_path) {
  folders <- list.dirs(algo_dir, full.names = TRUE, recursive = FALSE)
  
  for (seed_dir in folders) {
    metric_data <- read.csv(file.path(seed_dir, "final_model_metrics.csv"))
    combined_metric_data <- rbind(combined_metric_data,metric_data)
    
    data_testing_monitor <- data.frame()
    
    if (algo_dir == "C:\\Users\\sambu\\Downloads\\SWiMM_DEEPeR_old\\data") {
      data_testing_monitor <- read.csv(file.path(seed_dir, "inference.csv"))
      data_testing_monitor$Termination <- NULL
      data_testing_monitor$Episode <- NULL
      names(data_testing_monitor)[names(data_testing_monitor) == "Reward"] <- "r"
      names(data_testing_monitor)[names(data_testing_monitor) == "Length"] <- "l"
    }
    else {
      data_testing_monitor <- read.csv(file.path(seed_dir, "testing_monitor.csv"))
      data_testing_monitor$t <- NULL
    }
    
    data_testing_monitor$Algorithm <- metric_data$Algorithm[1:nrow(data_testing_monitor)]
    data_testing_monitor$Seed <- metric_data$Seed[1:nrow(data_testing_monitor)]
    
    combined_testing_montior <- rbind(combined_testing_montior,data_testing_monitor)
  }
}

monitor_agg <- aggregate(r ~ Algorithm, data = combined_testing_montior, FUN = mean, na.rm = TRUE)

agg_step <- aggregate(cbind(OutOfView, MaximumDistance, TargetCollision) ~ Algorithm + Seed + Episode, 
                      data = combined_metric_data, FUN = max, na.rm = TRUE)

sum_data <- aggregate(cbind(OutOfView, MaximumDistance, TargetCollision) ~ Algorithm, 
                      data = agg_step, FUN = sum, na.rm = TRUE)

# Normalise the safety errors
sum_data[, -which(names(sum_data) == "Algorithm")] <- sum_data[, -which(names(sum_data) == "Algorithm")] * length(unique(sum_data$Algorithm)) / nrow(agg_step)

mean_data <- aggregate(. ~ Algorithm,
                       data = combined_metric_data[, !(names(combined_metric_data) %in% c("Seed", "Episode", "Step", "OutOfView", "MaximumDistance", "TargetCollision"))],
                       FUN = mean, na.rm = TRUE)


# Normalising the azimuth and distance errors
SENSOR_WIDTH <- 4.98
SENSOR_HEIGHT <- 3.74
FOCAL_LENGTH <- 2.97

CAM_FOV <- 2 * atan(SENSOR_WIDTH / (2 * FOCAL_LENGTH)) * (180 / pi)

ALPHA <- CAM_FOV / 2.0

# Limit to 1 to get a suitable normalisation
mean_data$AError <- pmin(mean_data$AError / ALPHA, 1)
mean_data$DError <- mean_data$DError / 4

# Merge the results
combined_metric_data_summary <- merge(sum_data, mean_data, by = c("Algorithm"))
combined_metric_data_summary <- merge(combined_metric_data_summary, monitor_agg, by = c("Algorithm"))

df_long <- combined_metric_data_summary %>%
  pivot_longer(cols = -c(Algorithm, r),
               names_to = "Variable", values_to = "Value")

df_long$Algorithm <- factor(df_long$Algorithm, levels = c("SAC_OLD",
                                                          "SAC",
                                                          "PPO",
                                                          "TD3"))

df_long$Variable <- factor(df_long$Variable, levels = c("AError",
                                                        "DError",
                                                        "OutOfView",
                                                        "MaximumDistance",
                                                        "TargetCollision",
                                                        "ASmoothnessError",
                                                        "DSmoothnessError"))

increase <- 0

## Normal Version ##

# algo_labels <- c(
#   "SAC_OLD" = expression(SAC[paste("SWiMMv1.0")]),
#   "SAC" = expression(SAC[paste("SWiMMv2.0")]))

# ggplot(df_long, aes(x = r, y = Value, shape = Algorithm, color = Variable, group = Variable)) +
#   geom_point(size = 3) +
#   geom_line() +
#   scale_x_continuous(name = "Mean Episodic Reward", labels = scientific_10_wrapper(y = increase)) +
#   scale_y_continuous(name = "Value (Normalised)", labels = scientific_10_wrapper(y = increase)) +
#   scale_shape_manual(values = c(15, 16, 17, 18), labels = algo_labels) +
#   scale_color_brewer(palette = "Set2", name = "Metric", labels = c("AError" = "A",
#                                                                    "DError" = "D",
#                                                                    "OutOfView" = "A'",
#                                                                    "MaximumDistance" = "D'",
#                                                                    "TargetCollision" = "C'",
#                                                                    "ASmoothnessError" = expression(S[A]),
#                                                                    "DSmoothnessError" = expression(S[D]))) +
#   guides(color = guide_legend(nrow = 1),
#          shape = guide_legend(nrow = 1)) +
#   theme(legend.position = "bottom",
#         legend.direction = "horizontal",
#         legend.box = "vertical",
#         text = element_text(family = "Times New Roman"))

## ##

## Thesis Version ##

df_long <- df_long[df_long$Algorithm != "SAC_OLD", ]

ggplot(df_long, aes(x = r, y = Value, shape = Algorithm, color = Variable, group = Variable)) +
  geom_point(size = 3) +
  geom_line() +
  scale_x_continuous(name = "Mean Episodic Reward", labels = scientific_10_wrapper(y = increase)) +
  scale_y_continuous(name = "Value (Normalised)", labels = scientific_10_wrapper(y = increase)) +
  scale_shape_manual(values = c(15, 16, 17, 18)) +
  scale_color_brewer(palette = "Set2", name = "Metric", labels = c("AError" = "A",
                                                                   "DError" = "D",
                                                                   "OutOfView" = "A'",
                                                                   "MaximumDistance" = "D'",
                                                                   "TargetCollision" = "C'",
                                                                   "ASmoothnessError" = expression(S[A]),
                                                                   "DSmoothnessError" = expression(S[D]))) +
  guides(
    shape = guide_legend(nrow = 1, order = 1),
    color = guide_legend(nrow = 1, order = 2)
  ) +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        text = element_text(family = "Times New Roman"))

## ##

