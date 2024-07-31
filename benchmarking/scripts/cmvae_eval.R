library(utils)
library(data.table)
library(ggplot2)
library(yaml)
library(scales)

normalize_by_max <- function(df, group_cols, target_col) {
  max_values <- tapply(df[[target_col]], interaction(df[group_cols]), max)
  normalized_values <- df[[target_col]] / max_values[interaction(df[group_cols])]
  return(normalized_values)
}

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

directory_path <- list(
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\0\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\1\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\2\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\3\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\4\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\5\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\6\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\7\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\8\\best_model\\inference_results",
  "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae\\9\\best_model\\inference_results"
  )

combined_data <- data.frame()

for (seed_dir in directory_path) {
  seed <- file.path(dirname(dirname(seed_dir)), "configs", "env_config.yml")
  yaml_data <- as.data.frame(t(yaml.load_file(seed)))
  folders <- list.dirs(seed_dir, full.names = TRUE, recursive = FALSE)
  
  for (folder in folders) {
    data <- read.csv(file.path(folder, "prediction_stats.csv"))
    data <- subset(data, select = c(Feature, MAE, Standard.Error))
    
    data_img <- read.csv(file.path(folder, "prediction_img.csv"))
    
    data[nrow(data) + 1,] = c("Image", data_img$MAE, data_img$Standard.Error)
    data$MAE <- as.numeric(data$MAE)
    data$Standard.Error <- as.numeric(data$Standard.Error)
    
    data <- cbind(data, yaml_data$seed)
    
    data$seed <- factor(data$seed)
    names(data)[names(data) == "seed"] <- "ModelSeed"
    
    yaml_data1 <- as.data.frame(t(yaml.load_file(file.path(folder, "configs", "env_config.yml"))))
    data <- cbind(data, yaml_data1$seed)
    data$seed <- factor(data$seed)
    names(data)[names(data) == "seed"] <- "Seed"
    
    yaml_data2 <- as.data.frame(t(yaml.load_file(file.path(dirname(dirname(seed_dir)), "configs", "cmvae", "cmvae_training_config.yml"))))
    data$EarlyStopping <- ifelse(all(sapply(yaml_data2$window_size, is.numeric)), "Early Stopping", "No Early Stopping")
    data$EarlyStopping <- factor(data$EarlyStopping, levels = c("No Early Stopping", "Early Stopping"))
    
    combined_data <- rbind(combined_data,data)
  }
}

point_shape <- c(1:length(unique(factor(combined_data$Seed))))
point_shape <- point_shape[factor(combined_data$Seed)]

ggplot(combined_data, aes(x = EarlyStopping, y = MAE)) +
  geom_boxplot(position = position_dodge(1), outlier.shape = NA) +
  geom_jitter(position = position_dodge(1)) +
  facet_wrap(~ Feature, scales = "free_y") +
  scale_y_continuous(name = expression("MAE"), labels = scientific_10) +
  theme(legend.position = "bottom",
        axis.title.x = element_blank()) + 
  labs(fill = "Early Stopping") 

early_stop_arr <- c("No Early Stopping", "Early Stopping")

for (element in early_stop_arr) {
  section_data <- combined_data[combined_data$EarlyStopping == element, ]
  
  print(ggplot(section_data, aes(x = ModelSeed, y = MAE)) +
    geom_boxplot(position = position_dodge(1), outlier.shape = NA) +
    geom_jitter(aes(color=Seed)) +
    facet_wrap(~ Feature, scales = "free_y") +
    scale_y_continuous(name = expression("MAE"), labels = scientific_10) +
    theme(legend.position = "bottom")  + 
    labs(x = "Model Seed", color = "Inference Seed"))
}

combined_data$MAE <- normalize_by_max(combined_data, c("Feature"), "MAE")

combined_data <- aggregate(list(MAE = combined_data$MAE, Standard_Error = combined_data$Standard.Error),
                           by = list(ModelSeed = combined_data$ModelSeed,
                                     EarlyStopping = combined_data$EarlyStopping,
                                     Seed = combined_data$Seed),
                           FUN = mean)

