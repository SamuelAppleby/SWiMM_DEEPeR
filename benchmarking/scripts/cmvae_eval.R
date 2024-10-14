library(utils)
library(data.table)
library(ggplot2)
library(yaml)
library(scales)
library(extrafont)

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

directory_path <- "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\models\\cmvae"

combined_data_training <- data.frame()
combined_data <- data.frame()

for (seed_dir in list.dirs(directory_path, full.names = TRUE, recursive = FALSE)) {
  yaml_data <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "env_config.yml"))))
  yaml_data2 <- as.data.frame(t(yaml.load_file(file.path(seed_dir, "configs", "cmvae", "cmvae_training_config.yml"))))
  
  training_data <- read.csv(file.path(seed_dir, "run-.-tag-train_total_loss.csv"))
  training_data <- cbind(training_data, yaml_data$seed)
  training_data$WindowSize <- yaml_data2$window_size
  combined_data_training <- rbind(combined_data_training,training_data)
  
  for (instance_dir in list.dirs(file.path(seed_dir, 'best_model', 'inference_results'), full.names = TRUE, recursive = FALSE)) {
    data <- read.csv(file.path(instance_dir, "prediction_stats.csv"))
    data <- subset(data, select = c(Feature, MAE, Standard.Error))
    data$WindowSize <- yaml_data2$window_size
    
    data_img <- read.csv(file.path(instance_dir, "prediction_img.csv"))
    
    data[nrow(data) + 1,] = c("Image", data_img$MAE, data_img$Standard.Error, data$WindowSize)
    data$MAE <- as.numeric(data$MAE)
    data$Standard.Error <- as.numeric(data$Standard.Error)
    
    data <- cbind(data, yaml_data$seed)
    
    data$seed <- factor(data$seed)
    names(data)[names(data) == "seed"] <- "ModelSeed"
    
    yaml_data1 <- as.data.frame(t(yaml.load_file(file.path(instance_dir, "configs", "env_config.yml"))))
    data <- cbind(data, yaml_data1$seed)
    data$seed <- factor(data$seed)
    names(data)[names(data) == "seed"] <- "Seed"
    
    combined_data <- rbind(combined_data,data)
  }
}

combined_data$WindowSize[combined_data$WindowSize == 'NULL'] <- Inf
combined_data$WindowSize <- as.numeric(combined_data$WindowSize)

combined_data <- aggregate(list(MAE = combined_data$MAE, Standard_Error = combined_data$Standard.Error),
                           by = list(ModelSeed = combined_data$ModelSeed,
                                     WindowSize = combined_data$WindowSize,
                                     Feature = combined_data$Feature),
                           FUN = mean)

combined_data_training$WindowSize[combined_data_training$WindowSize == 'NULL'] <- Inf
combined_data_training$WindowSize <- as.numeric(combined_data_training$WindowSize)

combined_data_training <- aggregate(list(Wall.time = combined_data_training$Wall.time), 
                     by = list(ModelSeed = combined_data_training$seed,
                               WindowSize = combined_data_training$WindowSize),
                     FUN = function(i)max(i) - min(i))


combined_data <- merge(combined_data, combined_data_training, by = c("ModelSeed", "WindowSize"))
combined_data$WindowSize <- factor(combined_data$WindowSize)

combined_data$MAE[combined_data$Feature == "Image"] <- combined_data$MAE[combined_data$Feature == "Image"] / 255
combined_data$MAE[combined_data$Feature == "R"] <- combined_data$MAE[combined_data$Feature == "R"] / 4
combined_data$MAE[combined_data$Feature == "Theta"] <- combined_data$MAE[combined_data$Feature == "Theta"] / 360
combined_data$MAE[combined_data$Feature == "Psi"] <- combined_data$MAE[combined_data$Feature == "Psi"] / 360

combined_data$Feature <- factor(combined_data$Feature, levels = c("Image",
                                                        "R",
                                                        "Theta",
                                                        "Psi"))

shape_labels <- c(
  "11" = expression(paste("M"[CMVAE]^11, " / 5")),
  "13" = expression(paste("M"[CMVAE]^13, " / 5")),
  "17" = expression(paste("M"[CMVAE]^17, " / 5")),
  "19" = expression(paste("M"[CMVAE]^19, " / 5")),
  "23" = expression(paste("M"[CMVAE]^23, " / 5")),
  "47" = expression(paste("M"[CMVAE]^47, " / ", infinity)),
  "53" = expression(paste("M"[CMVAE]^53, " / ", infinity)),
  "59" = expression(paste("M"[CMVAE]^59, " / ", infinity)),
  "61" = expression(paste("M"[CMVAE]^61, " / ", infinity)),
  "67" = expression(paste("M"[CMVAE]^67, " / ", infinity))
)

ggplot(combined_data, aes(x = Wall.time, y = MAE, shape = ModelSeed, color = Feature, group = Feature)) +
  geom_point(size = 3) +
  geom_line() +
  scale_x_continuous(name = "Training Time (s)", labels = scientific_10) +
  scale_y_continuous(name = "MAE (Normalised)", labels = scientific_10) +
  scale_shape_manual(name = expression(paste("Model / ", omega)), values = c(0, 1, 2, 5, 10, 15, 16, 17, 18, 20), labels = shape_labels) +
  scale_color_brewer(palette = "Set2", name = "Feature", labels = c("Image" = "Image",
                                                                   "Psi" = expression(psi),
                                                                   "R" = "d",
                                                                   "Theta" = expression(theta))) +
  guides(color = guide_legend(nrow = 1, order = 1),
         shape = guide_legend(nrow = 2, order = 2)) +
  theme(legend.position = "bottom",
        legend.direction = "vertical",
        legend.box = "vertical",
        text = element_text(family = "Times New Roman"))
