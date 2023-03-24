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

# df_reward <- df_reward[df_reward$Step < 900000, ]


# Calculate average reward
df_reward$average_rew <- NA
df_reward[1,]$average_rew <- 0

window_size <- 100

for (i in seq(window_size, nrow(df_reward), by=window_size)) {
  # df_reward[i-99:i,]
  df_reward[i, ]$average_rew <- mean(df_reward[(i-window_size+1):i,]$Value)
}

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

ggplot(data=df_reward, aes(x=Step, y=average_rew)) +
  scale_x_continuous(name =TeX(r'(Step)'), labels = scientific_10)+ 
  scale_y_continuous(name =TeX(r'(Average Episodic Reward ($\mu=100$))'), labels = scientific_10)+ 
  geom_smooth(method = "loess", level=0.3) +
  # geom_smooth(method = lm) +
  # geom_smooth(method = "lm", formula = y ~ poly(x, 3), se = FALSE) +
  geom_point(aes(x=Step, y=Value, color=Length, shape=Termination)) +
  labs(color=TeX(r'(Steps per Episode)'), shape=TeX(r'(Termination Criteria)')) + 
  scale_color_gradient2(mid="#DC3220", high="#1A85FF", labels = scientific_10) +
  theme(legend.position="bottom") 


# spline.d <- as.data.frame(spline(df_reward$Step, df_reward$Value))
# 
# ggplot(data=spline.d, aes(x=Step, y=Value)) +
#   geom_line(data = spline.d, aes(x = x, y = y)) +
#   labs(x = TeX(r'($ |{S}| $)'), y = "Episodic Reward", color="Steps", shape="Termination") + 
#   theme(legend.position="bottom") 
