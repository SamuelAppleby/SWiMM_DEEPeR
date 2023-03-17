library(data.table)
library(reshape2)
library(ggplot2)
library(latex2exp)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_reward <- read.csv(file = 'C:/Users/sambu/Downloads/run-.-tag-episode_reward.csv')
df_termination <- read.csv(file = 'C:/Users/sambu/Downloads/run-.-tag-episode_termination.csv')
df_length <- read.csv(file = 'C:/Users/sambu/Downloads/run-.-tag-episode_length.csv')

names(df_termination)[names(df_termination) == 'Value'] <- 'Termination'
names(df_length)[names(df_length) == 'Value'] <- 'Length'

df_reward <- cbind(df_reward, df_termination[3])
df_reward <- cbind(df_reward, df_length[3])

df_reward$Termination <- cut(df_reward$Termination,
              breaks=c(-1, 0, 1),
              labels=c('Determine Over', 'Episode Max'))

ggplot(data=df_reward, aes(x=Step, y=Value)) +
  geom_smooth(method = "loess") +
  geom_point(aes(color=Length, shape=Termination)) +
  labs(x = TeX(r'($ |{S}| $)'), y = "Episodic Reward", color="Steps", shape="Termination Criteria") + 
  theme(legend.position="bottom") 


# spline.d <- as.data.frame(spline(df_reward$Step, df_reward$Value))
# 
# ggplot(data=spline.d, aes(x=Step, y=Value)) +
#   geom_line(data = spline.d, aes(x = x, y = y)) +
#   labs(x = TeX(r'($ |{S}| $)'), y = "Episodic Reward", color="Steps", shape="Termination") + 
#   theme(legend.position="bottom") 
