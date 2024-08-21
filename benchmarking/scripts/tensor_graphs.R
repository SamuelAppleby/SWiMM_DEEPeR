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

algos <- list("sac", "ppo")
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
    # time_training <- time_training + tail(data_training_monitor$t, 1)
    combined_data_training_time <- rbind(combined_data_training_time,data.frame(algo=algo, seed=yaml_data$seed, total_train_time=tail(data_training_monitor$t, 1)))
    
    data_test <- read.csv(file.path(seed_dir, "run-.-tag-eval_mean_ep_reward.csv"))
    names(data_test)[names(data_test) == "Value"] <- "TestingMeanEpisodeReward"
    data_test <- cbind(data_test, seed=yaml_data$seed)
    
    data_test$file_path <- seed_dir
    data_test$algo <- algo
    
    combined_data_test <- rbind(combined_data_test,data_test)
  }
}

# I can't do this dynamically due to the eval Python objects
# extension = "%s_1\\configs\\hyperparams\\%s.yml"
# formatted_path_algo <- sprintf(paste(formatted_path, extension, sep=""), algo, algo)
# yaml_data_algo <- as.data.frame(t(yaml.load_file(file.path(formatted_path_algo))))

algo_labels <- c("sac" = "SAC", "ppo" = "PPO", "td3" = "TD3")

# TRAINING REWARD GRAPH
ggplot(data=combined_data_training, aes(x=Step, y=TrainingMeanEpisodeReward, color=factor(seed))) +
  scale_x_continuous(name = "Step", labels = scientific_10)+
  scale_y_continuous(name =expression("Mean Episodic Reward ( " * mu * " = 10)"), labels = scientific_10) +
  facet_wrap(~ algo, labeller = labeller(algo = algo_labels)) +
  geom_smooth(aes(y=TrainingMeanEpisodeReward), method = "auto") +
  geom_point(aes(shape=Termination)) +
  labs(shape= "Termination Criteria") +
  scale_color_brewer(palette = "Set2", name = "Seed") +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))

# TRAINING TIME GRAPH
ggplot(data=combined_data_training_time, aes(x=reorder(algo, total_train_time), y=total_train_time, group=factor(seed), fill=factor(seed))) +
  geom_bar(stat="identity", position=position_dodge()) +
  scale_x_discrete(labels = algo_labels) +
  scale_y_continuous(name = "Total Training Time (s)", labels = scientific_10) +
  scale_fill_brewer(palette = "Set2", name = "Seed") +
  labs(x="Algorithm") +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))

sorted_df <- combined_data_test[order(combined_data_test$algo, -combined_data_test$TestingMeanEpisodeReward), ]
sorted_df <- sorted_df[!duplicated(sorted_df$algo), ]

format_annotation <- function(value) {
  sprintf("%.2e", value) %>% 
    gsub("e", "%*% 10^", .) %>% 
    parse(text = .)
}

sorted_df$StepLabel <- format_annotation(sorted_df$Step)
sorted_df$RewardLabel <- format_annotation(sorted_df$TestingMeanEpisodeReward)

# EVALUATION GRAPH
ggplot(data=combined_data_test, aes(x=Step, y=TestingMeanEpisodeReward, color=factor(seed))) +
  scale_x_continuous(name = "Step", labels = scientific_10)+
  scale_y_continuous(name = "Mean Episodic Reward", labels = scientific_10) +
  facet_wrap(~ algo, labeller = labeller(algo = algo_labels)) +
  geom_line(aes(y=TestingMeanEpisodeReward)) +
  geom_vline(data = sorted_df, aes(xintercept = Step), linetype = "dashed", color = "#66c2a5", inherit.aes = FALSE) +
  geom_text(data = sorted_df, aes(x = Step, y = Inf), 
            label = sorted_df$StepLabel, color = "#66c2a5", size = 4, fontface = "italic", 
            vjust = 1.5, hjust = 1.1, inherit.aes = FALSE) +
  geom_text(data = sorted_df, aes(x = Inf, y = TestingMeanEpisodeReward), 
            label = sorted_df$RewardLabel, color = "#66c2a5", size = 4, fontface = "italic", 
            vjust = 1.5, hjust = 1.1, inherit.aes = FALSE) +
  geom_hline(data = sorted_df, aes(yintercept = TestingMeanEpisodeReward), linetype = "dashed", color = "#66c2a5", inherit.aes = FALSE) +
  # annotate("text", x = Inf, y = sorted_df$TestingMeanEpisodeReward,
  #          label = format_annotation(sorted_df$TestingMeanEpisodeReward),
  #          hjust = 1.1, vjust = 1.5, color = "#66c2a5", size = 4, fontface = "italic") +
  scale_color_brewer(palette = "Set2", name = "Seed") +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))

# INFERENCE METRICS
# TODO I'll need to consider the best model from each algorithm
# combined_data_inference_summary <- data.frame(
#   algo = character(),
#   mean_inference_episodic_reward = numeric(),
#   total_inference_time = numeric()
# )
# 
# for (i in 1:nrow(sorted_df)) {
#   best_model <- sorted_df[i, ]
#   inference_dirs <- list.dirs(file.path(best_model$file_path, "inference"), full.names = TRUE, recursive = FALSE)
#   combined_data_inference <- data.frame()
#   
#   time_inference <- 0
#   for (inference_dir in inference_dirs) {
#     inference_data <- read.csv(file.path(inference_dir, "testing_monitor.csv"))
#     time_inference <- time_inference + tail(inference_data$t, 1)
#     yaml_data <- as.data.frame(t(yaml.load_file(file.path(inference_dir, "configs", "env_config.yml"))))
#     inference_data$seed <- as.numeric(yaml_data$seed)
#     inference_data$algo <- best_model$algo
#     combined_data_inference <- rbind(combined_data_inference,inference_data)
#   }
#   
#   mean_inference_episodic_reward <- mean(combined_data_inference$r, na.rm = TRUE)
#   
#   combined_data_inference_summary <- rbind(combined_data_inference_summary, data.frame(algo = best_model$algo,
#                                                                               mean_inference_episodic_reward = mean(combined_data_inference$r, na.rm = TRUE),
#                                                                               total_inference_time = time_inference))
# }
# 
# # combined_data_inference <- aggregate(list(r = combined_data_inference$r),
# #                                      by = list(seed = combined_data_inference$seed,
# #                                                algo = combined_data_inference$algo),
# #                                      FUN = mean)
# 
# ggplot(combined_data_inference, aes(x = algo, y = r)) +
#   geom_boxplot(position = position_dodge(1), outlier.shape = NA) +
#   geom_jitter(aes(color=factor(seed))) +
#   scale_x_discrete(name = "Algorithm", labels = algo_labels)+
#   scale_y_continuous(name = "Mean Episodic Reward", labels = scientific_10) +
#   scale_color_brewer(palette = "Set2", name = "Seed") +
#   theme(legend.position="bottom",text=element_text(family="Times New Roman"))


