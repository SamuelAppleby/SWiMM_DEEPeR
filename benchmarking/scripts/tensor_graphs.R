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

format_annotation <- function(value) {
  sprintf("%.2e", value) %>% 
    gsub("e", "%*% 10^", .) %>% 
    parse(text = .)
}

algo <- "sac"
base_path <- "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\%s\\"
formatted_path <- sprintf(base_path, algo)

folders <- list.dirs(formatted_path, full.names = TRUE, recursive = FALSE)

combined_data_training <- data.frame()
combined_data_test <- data.frame()

for (seed_dir in folders) {
  yaml_data <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "env_config.yml"))))
  yaml_data$seed <- factor(yaml_data$seed)
  
  data <- read.csv(file.path(seed_dir, "run-.-tag-rollout_ep_rew_mean.csv"))
  names(data)[names(data) == "Value"] <- "TrainingMeanEpisodeReward"
  data$Length <- c(data$Step[1], diff(data$Step))
  
  data_termination <- read.csv(file.path(seed_dir, "run-.-tag-rollout_episode_termination.csv"))
  names(data_termination)[names(data_termination) == "Value"] <- "Termination"
  
  data_termination$Termination <- factor(data_termination$Termination, 
                     levels = c(0, 1, 2, 3),  # Specify all possible numeric values
                     labels = c("Maximum Distance", "Target Out Of View", "Target Collision", "Threshold Reached"))
  
  data_termination$Termination <- as.character(data_termination$Termination)
  
  data$Termination <- data_termination$Termination
  data <- cbind(data, seed=yaml_data$seed)
  combined_data_training <- rbind(combined_data_training,data)
  
  data_test <- read.csv(file.path(seed_dir, "run-.-tag-eval_mean_ep_reward.csv"))
  names(data_test)[names(data_test) == "Value"] <- "TestingMeanEpisodeReward"
  data_test <- cbind(data_test, seed=yaml_data$seed)
  combined_data_test <- rbind(combined_data_test,data_test)
}

# I can't do this dynamically due to the eval Python objects
# extension = "%s_1\\configs\\hyperparams\\%s.yml"
# formatted_path_algo <- sprintf(paste(formatted_path, extension, sep=""), algo, algo)
# yaml_data_algo <- as.data.frame(t(yaml.load_file(file.path(formatted_path_algo))))

ggplot(data=combined_data_training, aes(x=Step, y=TrainingMeanEpisodeReward, color=seed)) +
  scale_x_continuous(name =expression("Step"), labels = scientific_10)+
  scale_y_continuous(name =expression("Mean Episodic Reward ( " * mu * " = 10)"), labels = scientific_10) +
  geom_smooth(aes(y=TrainingMeanEpisodeReward), method = "auto") +
  geom_point(aes(shape=Termination)) +
  labs(shape=expression("Termination Criteria")) +
  scale_color_brewer(palette = "Set2", name = "Seed") +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))


sorted_df <- combined_data_test[order(-combined_data_test$TestingMeanEpisodeReward), ]
best_model <- sorted_df[1, ]

ggplot(data=combined_data_test, aes(x=Step, y=TestingMeanEpisodeReward, color=seed)) +
  scale_x_continuous(name =expression("Step"), labels = scientific_10)+
  scale_y_continuous(name =expression("Mean Episodic Reward"), labels = scientific_10) +
  geom_line(aes(y=TestingMeanEpisodeReward)) +
  geom_vline(xintercept = best_model$Step, linetype = "dashed", color = "#66c2a5") +
  annotate("text", y = Inf, x = best_model$Step, 
           label = format_annotation(best_model$Step), 
           hjust = 1.1, vjust = 1.5, color = "#66c2a5", size = 4, fontface = "italic") +
  geom_hline(yintercept = best_model$TestingMeanEpisodeReward, linetype = "dashed", color = "#66c2a5") +
  annotate("text", x = Inf, y = best_model$TestingMeanEpisodeReward, 
           label = format_annotation(best_model$TestingMeanEpisodeReward), 
           hjust = 1.1, vjust = 1.5, color = "#66c2a5", size = 4, fontface = "italic") +
  scale_color_brewer(palette = "Set2", name = "Seed") +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))

# loadfonts(device="win")
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# 
# df_reward <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_reward.csv')
# df_termination <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_termination.csv')
# df_length <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_length.csv')
# 
# names(df_termination)[names(df_termination) == 'Value'] <- 'Termination'
# names(df_length)[names(df_length) == 'Value'] <- 'Length'
# 
# df_reward <- cbind(df_reward, df_termination[3])
# df_reward <- cbind(df_reward, df_length[3])
# 
# df_reward$Termination <- as.character(cut(df_reward$Termination,
#               breaks=c(-1, 0, 1),
#               labels=c('Determine Over', 'Episode Max')))
# 
# df_reward$average_rew <- NA
# window_size <- 100
# 
# for (i in seq(1, nrow(df_reward), by=1)) {
#   if(i <= 100){
#     df_reward[i, ]$average_rew <- mean(df_reward[1:i,]$Value)
#   }
#   else{
#     df_reward[i, ]$average_rew <- mean(df_reward[(i-window_size+1):i,]$Value)
#   }
# }
# 
# window_max <- df_reward[which.max(df_reward$average_rew),]
# window_min <- df_reward[which(df_reward$Step == window_max$Step) - window_size + 1,]
# best_region <- df_reward[which(df_reward$Step == window_min$Step):which(df_reward$Step == window_max$Step),]
# 
# rects <- data.frame(start=window_min$Step, end=window_max$Step, sliding_window=100)
# 
# ggplot(data=df_reward, aes(x=Step, y=Value)) +
#   scale_x_continuous(name =expression("Step"), labels = scientific_10)+
#   scale_y_continuous(name =expression("Episodic Reward"), labels = scientific_10)+
#   geom_smooth(aes(y=Value), method = "auto", color="black") +
#   geom_point(aes(color=Length, shape=Termination)) +
#   geom_rect(data=rects, inherit.aes=FALSE, aes(xmin=start, xmax=end, ymin=min(best_region$Value),
#            ymax=max(best_region$Value), fill=factor(sliding_window)), color="transparent", alpha=0.3) +
#   scale_fill_manual(expression(mu * " "),
#                     labels = expression(1%*%10^2),
#                     values = 'orange',
#                     guide = guide_legend(override.aes = list(alpha = 1))) +
#   labs(color=expression("Steps per Episode"), shape=expression("Termination Criteria")) +
#   scale_color_gradient2(mid="#000AFF", high="#00DDFF", labels = scientific_10) +
#   annotate("text", rects$start + (0.5 * (rects$end - rects$start)), y = max(best_region$Value) + 100,
#            label=as.character(expression(psi * " = " * 2.19%*%10^3)), parse=T) +
#   theme(legend.position="bottom",text=element_text(family="Times New Roman"))

