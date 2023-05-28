library(data.table)
library(reshape2)
library(ggplot2)
library(extrafont)
library(dplyr)

scientific_10 <- function(x) {
  parse(text=gsub("e", "%*% 10^", scales::scientific_format()(x)))
}

loadfonts(device="win")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_reward <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_reward.csv')
df_termination <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_termination.csv')
df_length <- read.csv(file = 'C:/Users/sambu/Downloads/2/run-.-tag-episode_length.csv')

names(df_termination)[names(df_termination) == 'Value'] <- 'Termination'
names(df_length)[names(df_length) == 'Value'] <- 'Length'

df_reward <- cbind(df_reward, df_termination[3])
df_reward <- cbind(df_reward, df_length[3])

df_reward$Termination <- as.character(cut(df_reward$Termination,
              breaks=c(-1, 0, 1),
              labels=c('Determine Over', 'Episode Max')))

df_reward$average_rew <- NA
window_size <- 100

for (i in seq(1, nrow(df_reward), by=1)) {
  if(i <= 100){
    df_reward[i, ]$average_rew <- mean(df_reward[1:i,]$Value)
  }
  else{
    df_reward[i, ]$average_rew <- mean(df_reward[(i-window_size+1):i,]$Value)
  }
}

window_max <- df_reward[which.max(df_reward$average_rew),]
window_min <- df_reward[which(df_reward$Step == window_max$Step) - window_size + 1,]
best_region <- df_reward[which(df_reward$Step == window_min$Step):which(df_reward$Step == window_max$Step),]

rects <- data.frame(start=window_min$Step, end=window_max$Step, sliding_window=100)

ggplot(data=df_reward, aes(x=Step, y=Value)) +
  scale_x_continuous(name =expression("Step"), labels = scientific_10)+
  scale_y_continuous(name =expression("Episodic Reward"), labels = scientific_10)+
  geom_smooth(aes(y=Value), method = "auto", color="black") +
  geom_point(aes(color=Length, shape=Termination)) +
  geom_rect(data=rects, inherit.aes=FALSE, aes(xmin=start, xmax=end, ymin=min(best_region$Value),
           ymax=max(best_region$Value), fill=factor(sliding_window)), color="transparent", alpha=0.3) +
  scale_fill_manual(expression(mu * " "),
                    labels = expression(1%*%10^2),
                    values = 'orange',
                    guide = guide_legend(override.aes = list(alpha = 1))) +
  labs(color=expression("Steps per Episode"), shape=expression("Termination Criteria")) +
  scale_color_gradient2(mid="#000AFF", high="#00DDFF", labels = scientific_10) +
  annotate("text", rects$start + (0.5 * (rects$end - rects$start)), y = max(best_region$Value) + 100,
           label=as.character(expression(psi * " = " * 2.19%*%10^3)), parse=T) +
  theme(legend.position="bottom",text=element_text(family="Times New Roman"))

