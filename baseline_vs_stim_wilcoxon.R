library(nlme)
library(multcomp)
library(ggplot2)
library(dplyr)
library(lsmeans)

setwd('G:/My Drive/Documents/PycharmProjects/OFC recording data/Data/Output')

df = read.csv('OFCRec_all_SU_AMsound_firing_rate.csv')

unit_list = c()
p_list = c()
v_list = c()
for (unit in unique(df$Unit)) {
  
  cur_unit_df = df[df$Unit == unit,]

  unit_list = c(unit_list, unit)
  wilcox = wilcox.test(FR_Hz ~ Period, paired=T, alternative="two.sided", data=cur_unit_df)
  p_list = c(p_list, wilcox$p.value)
  v_list = c(v_list, wilcox$statistic)

}

p_df = data.frame(Unit=unit_list, P_value=p_list, V_value=v_list)
write.csv(p_df, "OFCPL_SU_AMsound_wilcoxon.csv", row.names = F, append = F)
