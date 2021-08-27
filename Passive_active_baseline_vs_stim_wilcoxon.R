library(nlme)
library(multcomp)
library(ggplot2)
library(dplyr)
library(lsmeans)

setwd('G:/My Drive/Documents/PycharmProjects/OFC recording data/Data/Output')

df = read.csv('OFCRec_all_PSU_AMsound_firing_rate.csv')

unit_list = c()
p_list = c()
v_list = c()
cohen_list = c()
df$Period = factor(df$Period, levels=c('Baseline', 'Stimulus'))
for (unit in unique(df$Unit)) {
  
  cur_unit_df = df[df$Unit == unit,]

  unit_list = c(unit_list, unit)
  wilcox = wilcox.test(FR_Hz ~ Period, paired=T, alternative="two.sided", data=cur_unit_df)
  # wilcox = t.test(FR_Hz ~ Period, paired=T, data=cur_unit_df)
  p_list = c(p_list, wilcox$p.value)
  v_list = c(v_list, wilcox$statistic)
  cohen_list = c(cohen_list, effsize::cohen.d(FR_Hz ~ Period | Subject(Unit), paired=T,  data=cur_unit_df)$estimate)
}

p_df = data.frame(Unit=unit_list, P_value=p_list, V_value=v_list, Cohen_d = cohen_list)

write.csv(p_df, "OFCPL_PSU_AMsound_wilcoxon.csv", row.names = F, append = F)
