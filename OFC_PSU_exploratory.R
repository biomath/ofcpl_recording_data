library(nlme)
library(multcomp)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lsmeans)
library(glmmTMB)
library(DHARMa)
library(car)
library(ggpubr)
library(reshape2)
theme_set(theme_classic(base_size = 20))
# Plot variables
size_multiplier = 5
line_thickness = size_multiplier/5

setwd('G:/My Drive/Documents/PycharmProjects/OFC recording data/Data/Output')

z_score = function(response_df, baseline_df) {
  return(mean(response_df) - mean(baseline_df)) / sqrt(
    var(response_df) + var(baseline_df) - 
      2*cov(response_df, baseline_df))
}

df = read.csv('OFCRec_all_PSU_AMsound_firing_rate.csv')
# df_wilcoxon = read.csv('OFCPL_spoutOffset_wilcoxon.csv')
df = separate(data = df, col = Unit,  sep = "_", into=c("Subject", "Date","concat", "Cluster"), remove=F)
df$Session = ''
df[(grep('Aversive',df$Key_file)),]$Session = 'Aversive'
df[(grep('Pre', df$Key_file)),]$Session = 'Pre'
df[(grep('Post', df$Key_file)),]$Session = 'Post'
df$Subject_date = interaction(df$Subject, df$Date, sep="_")
df$Session = factor(df$Session, levels=c('Pre', 'Aversive', 'Post'))

# Add waveform measurements
wf_measurement_files = Sys.glob(file.path("../Waveform measurements/Passive-active/*waveform_measurements.csv"))

df$PTP_duration = 0
df$PTP_ratio = 0
df$Repolarization = 0
for (subject_date in unique(df$Subject_date)) {  # Subject_Date is unique
  cur_session_df = df[df$Subject_date == subject_date,]
  cur_file = wf_measurement_files[grep(subject_date, wf_measurement_files)]
  wf_measurements = read.csv(cur_file)
  for (cluster in unique(cur_session_df$Cluster)) {
    cluster_n = as.numeric(substr(cluster, 8, nchar(cluster)))
    df[(grepl(cluster, df$Unit)) & (df$Subject_date == subject_date), ]$PTP_duration = wf_measurements[wf_measurements$Cluster == cluster_n,]$PTP_duration_ms
    df[(grepl(cluster, df$Unit)) & (df$Subject_date == subject_date), ]$PTP_ratio = wf_measurements[wf_measurements$Cluster == cluster_n,]$PTP_ratio
    df[(grepl(cluster, df$Unit)) & (df$Subject_date == subject_date), ]$Repolarization = wf_measurements[wf_measurements$Cluster == cluster_n,]$Repolarization_duration_ms
  }
}


# Combine with behavioral performance
threshold_files = Sys.glob(file.path('G:/My Drive/Documents/PycharmProjects/OFC recording data/Data/Behavioral performance/psych_testing/*psychThreshold.csv'))

# threshold_df = df_zscore[,c('Subject', 'Session')]
# threshold_df = separate(data = threshold_df, col = Session,  sep = "-", into=c("Test_type", "Sound_type", "Date", "Time"), remove=F)

subj_list = c()
session_list = c()
threshold_list = c()
day_list = c()
df$Threshold = 0
for (subject in unique(df$Subject)) {
  cur_file = threshold_files[grep(subject, threshold_files)]
  cur_threshold_df = read.csv(cur_file)
  
  # Store all behavioral parameters
  subj_list = c(subj_list, rep(subject, nrow(cur_threshold_df)))
  session_list = c(session_list, cur_threshold_df$Block_id)
  threshold_list = c(threshold_list, cur_threshold_df$Threshold)
  day_list = c(day_list, 1:nrow(cur_threshold_df))
  # Pair with single-unit df
  for (session in cur_threshold_df$Block_id) {
    if(nrow(df[grep(session, df$Key_file), ]) > 0) {
      df[grep(session, df$Key_file), ]$Threshold = cur_threshold_df[cur_threshold_df$Block_id == session,]$Threshold
    }
  }
}

threshold_df = data.frame(Subject=subj_list, Session=session_list, Threshold=threshold_list, Day=day_list)
threshold_df$Day = as.numeric(threshold_df$Day)
threshold_df$Subject = factor(threshold_df$Subject)



# Group by Trial
df_grouped = df %>% group_by(Unit, Subject, Subject_date, Session, Period, PTP_duration, PTP_ratio, Repolarization) %>%
  summarise( 
    FR_Hz = mean(FR_Hz),
    Threshold = mean(Threshold)
  )


p =ggplot(data=df_grouped, aes(x=Session, y=FR_Hz, group = 1)) + geom_line(aes(group = Unit), alpha=0.1) +
  facet_wrap(.~Period) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Firing rate (Hz)') + xlab('Trial type') +
  theme(legend.position = "none")
p

p =ggplot(data=df_grouped, aes(x=Period, y=FR_Hz, group = 1)) + geom_line(aes(group = Unit), alpha=0.1) +
  facet_wrap(.~Session) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Firing rate (Hz)') + xlab('Trial type') +
  theme(legend.position = "none")
p


aversive_df = df_grouped[df_grouped$Session == 'Aversive',]


p = ggplot(data = aversive_df, aes(x = Threshold, y = FR_Hz)) +
  facet_grid(.~Period) + 
  geom_point(aes(color=Subject), size=5) +
  stat_cor(method = "pearson", label.x = -20, label.y = 20) +
  geom_smooth(method = "lm", se = T)
p


# Does the difference between aversive and pre correlate with performance?
df_preVsAversive = df[df$Period == 'Baseline',] %>% 
  group_by(Unit, Subject) %>%
  summarise( 
    Z_score = mean(z_score(FR_Hz[Session == 'Aversive'], FR_Hz[Session == 'Pre'])),
    Threshold = mean(Threshold[Session == 'Aversive'])  # Pre does not have a threshold
  )
df_preVsAversive = df_preVsAversive[complete.cases(df_preVsAversive),]
df_preVsAversive = df_preVsAversive[!is.infinite(df_preVsAversive$Z_score),]

p = ggplot(data = df_preVsAversive, aes(x = Threshold, y = Z_score)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  stat_cor(method = "pearson", label.x = -20, label.y = 20) +
  geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Spontaneous FR z-score (Aversive - Pre)') + xlab('Threshold')
p
ggsave(plot=p, 'OFCPL_preVsAversiveVsThreshold_baseline.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)


df_preVsAversive = df[df$Period == 'Stimulus',] %>% 
  group_by(Unit, Subject, AMdepth) %>%
  summarise( 
    Z_score = mean(z_score(FR_Hz[Session == 'Aversive'], FR_Hz[Session == 'Pre'])),
    Threshold = mean(Threshold[Session == 'Aversive'])  # Pre does not have a threshold
  )
df_preVsAversive = df_preVsAversive[complete.cases(df_preVsAversive),]
df_preVsAversive = df_preVsAversive[!is.infinite(df_preVsAversive$Z_score),]

p = ggplot(data = df_preVsAversive, aes(x = Threshold, y = Z_score)) +
  facet_wrap(.~factor(round(AMdepth, 2))) + 
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  stat_cor(method = "pearson", label.x = -20, label.y = 20) +
  geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Stimulus FR z-score (Aversive - Pre)') + xlab('Threshold')
p
ggsave(plot=p, 'OFCPL_preVsAversiveVsThreshold_baseline.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)






# Does the difference between aversive and post correlate with performance?
df_postVsAversive = df_grouped[df_grouped$Period == 'Baseline',] %>% 
  group_by(Unit, Subject, Subject_date, Period, PTP_duration, PTP_ratio, Repolarization) %>%
  summarise( 
    Z_score = z_score(FR_Hz[Session == 'Aversive'], FR_Hz[Session == 'Post']),
    Threshold = Threshold[Session == 'Aversive']  # Pre does not have a threshold
  )
df_postVsAversive = df_postVsAversive[complete.cases(df_postVsAversive),]
df_postVsAversive = df_postVsAversive[!is.infinite(df_postVsAversive$Z_score),]

p = ggplot(data = df_postVsAversive, aes(x = Threshold, y = Z_score)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  stat_cor(method = "pearson", label.x = -20, label.y = 20) + 
  geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Spontaneous FR z-score (Aversive - Post)') + xlab('Threshold')
p
ggsave(plot=p, 'OFCPL_preVsAversiveVsThreshold.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)


# Does the difference between pre and post correlate with performance?
df_postVsPre = df_grouped[df_grouped$Period == 'Baseline',] %>% 
  group_by(Unit, Subject, Subject_date, Period, PTP_duration, PTP_ratio, Repolarization) %>%
  summarise( 
    Z_score = z_score(FR_Hz[Session == 'Post'], FR_Hz[Session == 'Pre']),
    Threshold = Threshold[Session == 'Aversive']  # Pre does not have a threshold
  )
df_postVsPre = df_postVsPre[complete.cases(df_postVsPre),]
df_postVsPre = df_postVsPre[!is.infinite(df_postVsPre$Z_score),]

p = ggplot(data = df_postVsPre, aes(x = Threshold, y = Z_score)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  stat_cor(method = "pearson", label.x = -20, label.y = 20) + 
  geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Spontaneous FR z-score (Post - Pre)') + xlab('Threshold')
p
ggsave(plot=p, 'OFCPL_preVsAversiveVsThreshold.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)


# Does the FR before AM predict a hit?
df_grouped = df[df$Session == 'Aversive' & df$Period == 'Baseline',] %>% group_by(Unit, Subject, Subject_date, AMdepth) %>%
  summarise( 
    z_score_hitVsMiss = z_score(FR_Hz[Hit == 1], FR_Hz[Miss == 1]),
    Hit = mean(FR_Hz[Hit == 1]),
    Miss = mean(FR_Hz[Miss == 1]),
    Threshold = mean(Threshold)
  )
df_grouped = melt(df_grouped, measure.vars = c('Hit', 'Miss'), value.name = 'FR_Hz_baseline', variable.name='Trial_type')
df_grouped$Trial_type = factor(df_grouped$Trial_type)

p =ggplot(data=df_grouped, aes(x=Trial_type, y=FR_Hz_baseline, group = 1)) + 
  facet_wrap(.~factor(round(AMdepth, 2))) + 
  geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Preceding spontaneous firing rate (Hz)') + xlab('Trial type') +
  theme(legend.position = "none")
p

df_grouped = df[df$Session == 'Aversive' & df$Period == 'Baseline',] %>% group_by(Unit, Subject, Subject_date, AMdepth) %>%
  summarise( 
    z_score_hitVsMiss = z_score(FR_Hz[Hit == 1], FR_Hz[Miss == 1]),
    Threshold = mean(Threshold)
  )
p = ggplot(data = df_grouped, aes(x = AMdepth, y = z_score_hitVsMiss)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  # stat_cor(method = "pearson", label.x = -20, label.y = 20) + 
  # geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Spontaneous FR z-score (Aversive - Post)') + xlab('AM depth')
p


df_grouped = df[df$Session == 'Aversive' & df$Period == 'Stimulus',] %>% group_by(Unit, Subject, Subject_date, AMdepth) %>%
  summarise( 
    Hit = mean(FR_Hz[Hit == 1]),
    Miss = mean(FR_Hz[Miss == 1]),
    Threshold = mean(Threshold)
  )
df_grouped = melt(df_grouped, measure.vars = c('Hit', 'Miss'), value.name = 'FR_Hz_baseline', variable.name='Trial_type')
df_grouped$Trial_type = factor(df_grouped$Trial_type)

p =ggplot(data=df_grouped, aes(x=Trial_type, y=FR_Hz_baseline, group = 1)) + 
  facet_wrap(.~(round(AMdepth, 2))) +
  geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Trial firing rate (Hz)') + xlab('Trial type') +
  theme(legend.position = "none")
p


df_grouped = df[df$Session == 'Aversive' & df$Period == 'Stimulus',] %>% group_by(Unit, Subject, Subject_date, AMdepth) %>%
  summarise( 
    z_score_hitVsMiss = z_score(FR_Hz[Hit == 1], FR_Hz[Miss == 1]),
    Threshold = mean(Threshold)
  )
p =ggplot(data=df_grouped, aes(x=factor(round(AMdepth, 2)), y=z_score_hitVsMiss, group = 1)) + 
  geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Z-score (Hit - Miss)') + xlab('AM depth') +
  theme(legend.position = "none")
p

p = ggplot(data = df_grouped, aes(x = AMdepth, y = z_score_hitVsMiss)) +
  # facet_wrap(.~(round(AMdepth, 2))) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  # stat_cor(method = "pearson", label.x = -20, label.y = 20) + 
  # geom_hline(yintercept=0, linetype="dashed", color = "red", size=line_thickness) +
  ylab('Spontaneous FR z-score (Aversive - Post)') + xlab('Threshold')
p


# df_zscore = df %>% group_by(Unit, Subject, Session, AMdepth, Hit, Miss, CR, FA, PTP_duration, PTP_ratio, Repolarization) %>%
#   summarise( 
#     Z_score = (mean(FR_Hz[Period == 'Stimulus']) - mean(FR_Hz[Period == 'Baseline'])) / sqrt(
#       var(FR_Hz[Period == 'Stimulus']) + var(FR_Hz[Period == 'Baseline']) - 2*cov(FR_Hz[Period == 'Stimulus'], FR_Hz[Period == 'Baseline']))
#   )

df_zscore = df %>% group_by(Unit, Subject, Subject_date, Session, PTP_duration, PTP_ratio, Repolarization, Date) %>%
  summarise( 
    Z_score = z_score(FR_Hz[Period == 'Stimulus'], FR_Hz[Period == 'Baseline']),
    d_prime = 2*(mean(FR_Hz[Period == 'Stimulus']) - mean(FR_Hz[Period == 'Baseline'])) / (sd(FR_Hz[Period == 'Stimulus']) + sd(FR_Hz[Period == 'Baseline']))
    )

df_zscore = df_zscore[complete.cases(df_zscore),]
df_zscore = df_zscore[!is.infinite(df_zscore$Z_score),]
# p = ggplot(df_zscore, aes(x=Z_score)) +
#   geom_histogram(bins=100)
# p
# ggsave(plot=p, 'OFCPL_offspoutZscore_histogram.pdf', width=15, height=10,dpi=600, useDingbats=FALSE)

p =ggplot(data=df_zscore, aes(x=Session, y=Z_score, group = 1)) + geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Z score') + xlab('Trial type')
p

p =ggplot(data=df_zscore, aes(x=Session, y=Z_, group = 1)) + geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('Z score') + xlab('AMdepth')
p

p =ggplot(data=df_zscore, aes(x=Session, y=d_prime, group = 1)) + geom_line(aes(group = Unit), alpha=0.1) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.5) +
  ylab('d\'') + xlab('AMdepth')
p


df_zscore$Subject = factor(df_zscore$Subject)
df_zscore$Unit = factor(df_zscore$Unit)
# lme_bin <- glmmTMB(Z_score~TType*PTP_duration + (1|Subject/Unit), data=df_zscore, family=gaussian)
# simulationOutput <- simulateResiduals(fittedModel = lme_bin, plot = T, re.form=NULL)  # pass
# Anova(lme_bin)
# pairs(lsmeans(lme_bin,  ~ TType), adjust='Tukey')


# Behavior plot only up to day 7 (before losing subjects) on log scale
p = ggplot(data = threshold_df[threshold_df$Day %in% 1:7,], aes(x = Day, y = Threshold, group=1)) +
  stat_summary(fun.y="mean", geom="point", na.rm = T, color='Black', alpha = 1, size=line_thickness*3) +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='Black', na.rm = T, size=line_thickness/2, width=0.05) +
  geom_point(aes(color=Subject), size=5) +
  stat_smooth(geom="line", method = "lm", se = F, aes(color=Subject, group=Subject), alpha=1, size=line_thickness) +
  # stat_smooth(geom="line", method = "lm", se = T, aes(group=1), alpha=0.8, size=line_thickness*3) +
  scale_x_continuous(trans = "log10")
p
ggsave(plot=p, 'OFCPL_learningThresholds.pdf', width=15, height=10,dpi=600, useDingbats=FALSE)


# Plot baseline firing by threshold
df_zscore$Threshold = 100
for(subject in unique(df_zscore$Subject)) {
  for(date in unique(df_zscore[df_zscore$Subject == subject,]$Date)) {
    cur_threshold = threshold_df[threshold_df$Subject == subject & (grepl(date, threshold_df$Session)),]$Threshold
    df_zscore[df_zscore$Subject == subject & df_zscore$Date == date,]$Threshold = cur_threshold
  }
}

df_zscore[df_zscore$Threshold == 0,]$Threshold = NA
df_zscore = df_zscore[complete.cases(df_zscore),]

p = ggplot(data = df_zscore, aes(x = d_prime, y = Threshold)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  geom_vline(xintercept=0, linetype="dashed", color = "red", size=line_thickness)
p
ggsave(plot=p, 'OFCPL_zscoreVsThreshold.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)


df_grouped[df_grouped$Threshold == 0,]$Threshold = NA
df_zscore = df_zscore[complete.cases(df_zscore),]

p = ggplot(data = df_zscore, aes(x = Z_score, y = Threshold)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T) +
  geom_vline(xintercept=0, linetype="dashed", color = "red", size=line_thickness)
p
ggsave(plot=p, 'OFCPL_zscoreVsThreshold.pdf', width=10, height=10,dpi=600, useDingbats=FALSE)


# pairs(lsmeans(lme_bin,  ~ Threshold), adjust='Tukey')


##### Waveform measurements
waveform_df = df %>% group_by(Unit, Subject) %>%
  summarise(
    PTP_duration = mean(PTP_duration),
    PTP_ratio = mean(PTP_ratio),
    Repolarization = mean(Repolarization),
    Baseline_FR = mean(FR_Hz[Period=="Baseline"]),
    Stimulus_FR = mean(FR_Hz[Period=="Stimulus"])
  )
waveform_df$Overall_FR = (waveform_df$Baseline_FR + waveform_df$Stimulus_FR)/2

p = ggplot(data = waveform_df, aes(x = PTP_duration, y = 1/PTP_ratio)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T)
p
ggsave(plot=p, 'OFCPL_PTPdurationVsRatio.pdf', width=15, height=10,dpi=600, useDingbats=FALSE)

p = ggplot(data = waveform_df, aes(x = PTP_duration, y = Repolarization)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T)
p

p = ggplot(data = waveform_df, aes(x = PTP_duration, y = Overall_FR)) +
  geom_point(aes(color=Subject), size=5) +
  geom_smooth(method = "lm", se = T)
p
ggsave(plot=p, 'OFCPL_PTPdurationVsFR.pdf', width=15, height=10,dpi=600, useDingbats=FALSE)

hist(waveform_df$PTP_duration, breaks=50)
