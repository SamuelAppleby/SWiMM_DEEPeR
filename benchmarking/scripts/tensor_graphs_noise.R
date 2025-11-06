library(data.table)
library(reshape2)
library(ggplot2)
library(extrafont)
library(dplyr)
library(utils)
library(data.table)
library(yaml)
library(scales)

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

custom_aggregate <- function(df) {
  # Number of rows in the new dataframe
  new_row_count <- nrow(df) / 2
  
  # Initialize an empty dataframe to store the results
  result <- data.frame(matrix(ncol = ncol(df), nrow = new_row_count))
  colnames(result) <- colnames(df)
  
  # Loop through and calculate the average for the first column and max for the second column
  for (i in 1:new_row_count) {
    result[i, 1] <- mean(df[(2*i-1):(2*i), 1])  # Average of the first column
    result[i, 2] <- max(df[(2*i-1):(2*i), 2])   # Maximum of the second column
  }
  
  return(result)
}

algos <- list("sac", "sac_noise")

combined_data_training <- data.frame()
combined_data_test <- data.frame()
combined_data_training_time <- data.frame(
  algo = character(),
  seed = numeric(),
  total_train_time = numeric()
)

for (algo in algos) {
  base_path <- "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\%s\\"
  formatted_path <- sprintf(base_path, algo)
  
  folders <- list.dirs(formatted_path, full.names = TRUE, recursive = FALSE)
  
  for (seed_dir in folders) {
    yaml_data <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "env_config.yml"))))
    
    if (as.numeric(yaml_data$seed) != 97) {
      next
    }
    
    data_training <- read.csv(file.path(seed_dir, "run-.-tag-rollout_ep_rew_mean.csv"))
    names(data_training)[names(data_training) == "Value"] <- "TrainingMeanEpisodeReward"
    data_training$Length <- c(data_training$Step[1], diff(data_training$Step))
    
    data_termination <- read.csv(file.path(seed_dir, "run-.-tag-rollout_episode_termination.csv"))
    names(data_termination)[names(data_termination) == "Value"] <- "Termination"
    
    data_termination$Termination <- factor(data_termination$Termination, 
                                           levels = c(0, 1, 2, 3),
                                           labels = c("Maximum Distance", "Target Out Of View", "Target Collision", "Threshold Reached"))
    
    data_termination$Termination <- as.character(data_termination$Termination)
    
    data_training$Termination <- data_termination$Termination
    data_training$seed <- as.numeric(yaml_data$seed)
    data_training$algo <- algo
    combined_data_training <- rbind(combined_data_training,data_training)
    
    data_training_monitor <- read.csv(file.path(seed_dir, "training_monitor.csv"))
    combined_data_training_time <- rbind(combined_data_training_time,data.frame(algo=algo, seed=yaml_data$seed, total_train_time=tail(data_training_monitor$t, 1)))
    
    data_test <- read.csv(file.path(seed_dir, "run-.-tag-eval_mean_ep_reward.csv"))
    
    names(data_test)[names(data_test) == "Value"] <- "TestingMeanEpisodeReward"
    data_test <- cbind(data_test, seed=yaml_data$seed)
    
    data_test$file_path <- seed_dir
    data_test$algo <- algo
    
    # This is grabbing the testing_monitor.csv file to obtain the time passed since training (so that we know the running time for each evaluation)
    data_testing_monitor <- read.csv(file.path(seed_dir, "testing_monitor.csv"))
    data_testing_monitor <- subset(data_testing_monitor, select = -l)
    data_testing_monitor <- custom_aggregate(data_testing_monitor)
    names(data_testing_monitor)[names(data_testing_monitor) == "r"] <- "TestingMeanEpisodeReward"
    data_test$time_for_best_model <- data_testing_monitor$t
    
    combined_data_test <- rbind(combined_data_test,data_test)
  }
}

# I can't do this dynamically due to the eval Python objects
# extension = "%s_1\\configs\\hyperparams\\%s.yml"
# formatted_path_algo <- sprintf(paste(formatted_path, extension, sep=""), algo, algo)
# yaml_data_algo <- as.data.frame(t(yaml.load_file(file.path(formatted_path_algo))))

## Normal Version ##

# algo_labels <- c(
#   "sac" = expression(SAC[paste("SWiMMv2.0")]^{"noiseless"}),
#   "sac_noise" = expression(SAC[paste("SWiMMv2.0")]^{"noisy"})
# )

## ##

## Thesis Version ##

algo_labels <- c(
  "sac" = expression(SAC[paste("noiseless")]),
  "sac_noise" = expression(SAC[paste("noisy")])
)

## ##

combined_data_training$algo <- factor(combined_data_training$algo, levels = c("sac", "sac_noise"))

# TRAINING REWARD GRAPH
ggplot(data = combined_data_training, aes(x = Step, y = TrainingMeanEpisodeReward, color = algo)) +
  scale_x_continuous(name = "Step", labels = scientific_10) +
  scale_y_continuous(name = expression("Mean Episodic Reward (" * mu * " = 10)"), labels = scientific_10) +
  geom_smooth(aes(y = TrainingMeanEpisodeReward), method = "auto") +
  geom_point(aes(shape = Termination)) +
  labs(shape = "Termination Criteria") +
  scale_color_brewer(palette = "Set2", name = "Environment", labels = algo_labels) +
  guides(
    shape = guide_legend(nrow = 1, order = 1),
    color = guide_legend(nrow = 1, order = 2)
    ) +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        text = element_text(family = "Times New Roman"))

sorted_df <- combined_data_test[order(combined_data_test$algo, -combined_data_test$TestingMeanEpisodeReward), ]
sorted_df <- sorted_df[!duplicated(sorted_df[c("algo", "seed")]), ]
sorted_df$algo <- factor(sorted_df$algo, levels = c("sac", "sac_noise"))

combined_data_training_time <- merge(combined_data_training_time, sorted_df, by = c("algo", "seed"))

long_data <- melt(combined_data_training_time,
                  id.vars = c("algo", "seed"),
                  measure.vars = c("total_train_time", "time_for_best_model"),
                  variable.name = "type",
                  value.name = "value")

long_data$algo <- factor(long_data$algo, levels = c("sac", "sac_noise"))

timings_for_paper <- aggregate(value ~ algo + type, data = long_data, FUN = mean, na.rm = TRUE)

# TRAINING TIME GRAPH
ggplot(data = long_data, aes(x = algo, y = value)) +
  geom_col(data = filter(long_data, type == "total_train_time"),
           position = position_dodge(width = 0.9),
           alpha = 0.4) +
  geom_col(data = filter(long_data, type == "time_for_best_model"),
           position = position_dodge(width = 0.9),
           alpha = 1) +
  scale_x_discrete(labels = algo_labels) +
  scale_y_continuous(name = "Time (s)", labels = scientific_10) +
  scale_fill_brewer(palette = "Set2", name = "Seed") +
  labs(x = "Environment") +
  geom_tile(aes(y=NA_integer_, alpha = factor(type))) +
  scale_alpha_manual(values = c(`total_train_time` = 0.4, `time_for_best_model` = 1),
                     labels = c("Total Train Time", "Best Model Found"),
                     name = "Time Type") +
  theme(legend.position = "bottom", text = element_text(family = "Times New Roman"))

format_annotation <- function(value) {
  sprintf("%.2e", value) %>%
    gsub("e", "%*% 10^", .) %>%
    parse(text = .)
}

sorted_df <- sorted_df[!duplicated(sorted_df$algo), ]
sorted_df$StepLabel <- format_annotation(sorted_df$Step)
sorted_df$RewardLabel <- format_annotation(sorted_df$TestingMeanEpisodeReward)

combined_data_test$algo <- factor(combined_data_test$algo, levels = c("sac", "sac_noise"))

# EVALUATION GRAPH
ggplot(data=combined_data_test, aes(x=Step, y=TestingMeanEpisodeReward, color=algo)) +
  scale_x_continuous(name = "Step", labels = scientific_10)+
  scale_y_continuous(name = "Mean Episodic Reward", labels = scientific_10) +
  geom_line(aes(y=TestingMeanEpisodeReward)) +
  geom_vline(data = sorted_df, aes(xintercept = Step, color = algo), linetype = "dashed", inherit.aes = FALSE, show.legend=FALSE) +
  geom_text(data = sorted_df, aes(x = Step, y = Inf, color = algo),
            label = sorted_df$StepLabel, size = 4, fontface = "italic",
            hjust = 1.2, vjust = 1.2, inherit.aes = FALSE, show.legend=FALSE) +
  geom_hline(data = sorted_df, aes(yintercept = TestingMeanEpisodeReward, color = algo), linetype = "dashed", inherit.aes = FALSE, show.legend=FALSE) +
  geom_text(data = sorted_df, aes(x = -Inf, y = TestingMeanEpisodeReward, color = algo),
            label = sorted_df$RewardLabel, size = 4, fontface = "italic",
            hjust = 1.1, inherit.aes = FALSE, show.legend=FALSE) +
  scale_color_brewer(palette = "Set2", name = "Environment", labels = algo_labels) +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))
