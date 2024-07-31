library(utils)
library(data.table)
library(ggplot2)
library(yaml)
library(scales)

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
  
  data$EarlyStopping <- ifelse(all(sapply(training_data$window_size, is.numeric)), "Early Stopping", "No Early Stopping")
  data$EarlyStopping <- factor(data$EarlyStopping, levels = c("No Early Stopping", "Early Stopping"))
  
  combined_data <- rbind(combined_data,data)
}

timings <- aggregate(list(Wall.time = combined_data$Wall.time), 
                  by = list(ModelSeed = combined_data$ModelSeed,
                            EarlyStopping = combined_data$EarlyStopping),
                  FUN = function(i)max(i) - min(i))

ggplot(timings, aes(x = EarlyStopping, y = Wall.time)) +
  geom_boxplot(position = position_dodge(1), outlier.shape = NA) +
  geom_jitter(aes(color=ModelSeed)) +
  theme(legend.position = "bottom",
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank())  + 
  scale_y_continuous(name =expression("Training Time(s)"), labels = scientific_10)+
  labs(x = "Early Stopping", y = "Training Time (s)")

early_stop_arr <- c("No Early Stopping", "Early Stopping")

for (element in early_stop_arr) {
  early_data <- subset(combined_data, EarlyStopping == element)
  
  print(ggplot(data = early_data, aes(x = Step, y = TrainingLoss, colour = ModelSeed)) +
    geom_line() +
    scale_x_continuous(name =expression("Epoch"), labels = scientific_10)+
    scale_y_continuous(name =expression("Training Loss"), labels = scientific_10)+
    theme(legend.position = "bottom"))
  
  print(ggplot(data = early_data, aes(x = Step, y = ValidationLoss, colour = ModelSeed)) +
    geom_line() +
    scale_x_continuous(name =expression("Epoch"), labels = scientific_10)+
    scale_y_continuous(name =expression("Validation Loss"), labels = scientific_10)+
    theme(legend.position = "bottom"))
}

timings <- aggregate(list(Wall.time = timings$Wall.time), 
                     by = list(EarlyStopping = timings$EarlyStopping),
                     FUN = mean)

