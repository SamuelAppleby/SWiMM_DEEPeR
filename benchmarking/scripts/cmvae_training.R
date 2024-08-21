library(utils)
library(data.table)
library(ggplot2)
library(yaml)
library(scales)
library(extrafont)

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

folders <- list.dirs("C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\", full.names = TRUE, recursive = FALSE)

combined_data <- data.frame()

for (seed_dir in folders) {
  yaml_data <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "env_config.yml"))))

  data <- read.csv(file.path(seed_dir, "run-.-tag-train_total_loss.csv"))
  names(data)[names(data) == "Value"] <- "TrainingLoss"
  
  data_test <- read.csv(file.path(seed_dir, "run-.-tag-test_total_loss.csv"))
  names(data_test)[names(data_test) == "Value"] <- "ValidationLoss"
  
  common_cols <- intersect(names(data), names(data_test))
  
  data <- data[, c(common_cols, "TrainingLoss")]
  data_test <- data_test[, c(common_cols, "ValidationLoss")]
  
  data <- cbind(data, ValidationLoss = data_test$ValidationLoss)
  
  data <- cbind(data, yaml_data$seed)
  
  data$seed <- factor(data$seed)
  names(data)[names(data) == "seed"] <- "ModelSeed"
  
  training_data <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "cmvae", "cmvae_training_config.yml"))))
  
  data$EarlyTermination <- ifelse(all(sapply(training_data$window_size, is.numeric)), "Early Termination", "No Early Termination")
  data$EarlyTermination <- factor(data$EarlyTermination, levels = c("No Early Termination", "Early Termination"))
  
  combined_data <- rbind(combined_data,data)
}

timings <- aggregate(list(Wall.time = combined_data$Wall.time), 
                  by = list(ModelSeed = combined_data$ModelSeed,
                            EarlyTermination = combined_data$EarlyTermination),
                  FUN = function(i)max(i) - min(i))

ggplot(timings, aes(x = EarlyTermination, y = Wall.time)) +
  geom_boxplot(position = position_dodge(1), outlier.shape = NA) +
  geom_jitter(aes(color=ModelSeed)) +
  scale_color_brewer(palette = "Set2", name = "Seed") +
  theme(legend.position = "bottom",
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        text=element_text(family="Times New Roman"))  + 
  scale_y_continuous(name = "Training Time(s)", labels = scientific_10) +
  scale_x_discrete(labels = c("No Early Termination" = expression(w == infinity), 
                              "Early Termination" = expression(w == 5))) +
  labs(x = "Early Termination", y = "Training Time (s)")

early_stop_arr <- c("No Early Termination", "Early Termination")

for (element in early_stop_arr) {
  early_data <- subset(combined_data, EarlyTermination == element)
  
  print(ggplot(data = early_data, aes(x = Step, y = TrainingLoss, colour = ModelSeed)) +
    geom_line() +
    scale_x_continuous(name = "Epoch", labels = scientific_10)+
    scale_y_continuous(name = "Training Loss", labels = scientific_10)+
    scale_color_brewer(palette = "Set2", name = "Seed") +
    theme(legend.position = "bottom",
          text=element_text(family="Times New Roman")))
  
  print(ggplot(data = early_data, aes(x = Step, y = ValidationLoss, colour = ModelSeed)) +
    geom_line() +
    scale_x_continuous(name = "Epoch", labels = scientific_10)+
    scale_y_continuous(name = "Validation Loss", labels = scientific_10)+
    scale_color_brewer(palette = "Set2", name = "Seed") +
    theme(legend.position = "bottom",
          text=element_text(family="Times New Roman")))
}

timings <- aggregate(list(Wall.time = timings$Wall.time), 
                     by = list(EarlyTermination = timings$EarlyTermination),
                     FUN = mean)

