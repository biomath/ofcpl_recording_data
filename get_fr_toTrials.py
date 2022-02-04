from intrastim_correlation import *
from pandas import read_csv
from os.path import sep
import platform

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def get_fr_toTrials(memory_name,
                    key_path_info,
                    key_path_spout,
                    unit_name,
                    output_path,
                    cur_unitData,
                    csv_pre_name=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    baseline_duration_for_fr_s=0.5,
                    stim_duration_for_fr_s=0.5,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    afterTrial_FR_start=1.3,  # For calculating after-stimulus firing rate; useful for Misses
                    afterTrial_FR_end=2):
    """
    Process spike data around trials
    Takes key files and writes two files:
    1. Firing rates within window (_AMsound_firing_rate.csv)
    2. Number of spout events during that window (_AMsound_firing_rate.csv)
    3. All timestamped spikes during that window for timeseries analyses (_AMsound_spikes.json)
    """
    # Load key files
    info_key_times = read_csv(key_path_info)
    spout_key_times = read_csv(key_path_spout)

    # Load spike times
    spike_times = np.genfromtxt(memory_name)

    # The default is 0, so this will only make a difference if set at function call
    # breakpoint_offset_time = breakpoint_offset / sampling_rate # OBSOLETE: offsets are now in seconds
    breakpoint_offset_time = breakpoint_offset

    # Grab GO trials and FA trials for stim response then walk back to get the immediately preceding CR trial
    # One complicating factor is spout onset/offset but I'll ignore this for now
    # Also ignore reminder trials
    relevant_key_times = info_key_times[
        ((info_key_times['TrialType'] == 0) | (info_key_times['FA'] == 1)) & (info_key_times['Reminder'] == 0)]

    # Now grab spike times
    # Baseline will be the CR trial immediately preceding a GO or a FA
    nonAM_FR_list = list()
    trial_FR_list = list()
    afterTrial_FR_list = list()
    nonAM_spikeCount_list = list()
    trial_spikeCount_list = list()
    afterTrial_spikeCount_list = list()
    previous_nonAM_FR_list = list()
    previous_trial_FR_list = list()
    previous_afterTrial_FR_list = list()
    previous_nonAM_spikeCount_list = list()
    previous_trial_spikeCount_list = list()
    previous_afterTrial_spikeCount_list = list()
    nonAM_spoutOn_frequency_list = list()
    trial_spoutOn_frequency_list = list()
    postStim_spoutOn_frequency_list = list()
    nonAM_spoutOff_frequency_list = list()
    trial_spoutOff_frequency_list = list()
    postStim_spoutOff_frequency_list = list()
    spoutOn_timestamps_list = list()
    spoutOff_timestamps_list = list()
    timestamped_trial_spikes = list()
    for _, cur_trial in relevant_key_times.iterrows():
        # Get spike times around the current stimulus onset
        # For baseline, go to the previous trial that resulted in a correct rejection (NO-GO) which
        # may not be the immediately preceding trial
        try:
            previous_cr = info_key_times[(info_key_times['CR'] == 1) &
                                         (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
        except IndexError:  # In case there is no CR before a hit, skip trial
            continue

        try:
            previous_hitOrMiss = info_key_times[((info_key_times['Hit'] == 1) | (info_key_times['Miss'] == 1)) &
                                                (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
            try:
                previous_previous_cr = \
                    info_key_times[(info_key_times['CR'] == 1) &  # To calculate the previous trial nonAM firing
                                       (info_key_times['TrialID'] < previous_hitOrMiss['TrialID'])].iloc[-1]
            except IndexError:  # In case there is none NaN it
                previous_previous_cr = None

        except IndexError:  # In case there is none NaN it
            previous_hitOrMiss = None
            previous_previous_cr = None


        # Count the number of spout onsets and offsets during the current trials; could be interesting...
        # Also transform to Hz in case I decide to change the window in the future
        nonAM_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 previous_cr['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 previous_cr['Trial_onset'] + baseline_duration_for_fr_s)]['Spout_onset'].values
        nonAM_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 previous_cr['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 previous_cr['Trial_onset'] + baseline_duration_for_fr_s)]['Spout_offset'].values

        trial_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + stim_duration_for_fr_s)]['Spout_onset'].values
        trial_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + stim_duration_for_fr_s)]['Spout_offset'].values

        afterTrial_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset'] + afterTrial_FR_start) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + afterTrial_FR_end)]['Spout_onset'].values
        afterTrial_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset'] + afterTrial_FR_start) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + afterTrial_FR_end)]['Spout_offset'].values

        # DEBUGGING:
        # if np.sum([baseline_spoutOn, baseline_spoutOff, stim_spoutOn, stim_spoutOff]) > 1:
        #     print()

        # Append to lists
        nonAM_spoutOn_frequency_list.append(len(nonAM_spoutOn) / baseline_duration_for_fr_s)
        nonAM_spoutOff_frequency_list.append(len(nonAM_spoutOff) / baseline_duration_for_fr_s)
        trial_spoutOn_frequency_list.append(len(trial_spoutOn) / stim_duration_for_fr_s)
        trial_spoutOff_frequency_list.append(len(trial_spoutOff) / stim_duration_for_fr_s)
        postStim_spoutOn_frequency_list.append(len(afterTrial_spoutOn) / (afterTrial_FR_end - afterTrial_FR_start))
        postStim_spoutOff_frequency_list.append(len(afterTrial_spoutOff) / (afterTrial_FR_end - afterTrial_FR_start))

        # Spout events around AM trial [trial_onset - baseline_duration; trial_onset + stim_duration]
        spoutOn_around_trial = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset'] - pre_stim_raster) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + post_stim_raster)]['Spout_onset'].values
        spoutOff_around_trial = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset'] - pre_stim_raster) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + post_stim_raster)]['Spout_offset'].values
        # zero-align to trial onset
        spoutOn_around_trial -= (cur_trial['Trial_onset'])
        spoutOff_around_trial -= (cur_trial['Trial_onset'])
        spoutOn_timestamps_list.append(spoutOn_around_trial)
        spoutOff_timestamps_list.append(spoutOff_around_trial)

        # Now get the spikes
        nonAM_spikes = spike_times[
            (previous_cr['Trial_onset'] + breakpoint_offset_time < spike_times) &
            (spike_times < previous_cr['Trial_onset'] + breakpoint_offset_time + baseline_duration_for_fr_s)]

        trial_spikes = spike_times[(spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time) &
                                   (spike_times < (cur_trial[
                                                       'Trial_onset'] + breakpoint_offset_time + stim_duration_for_fr_s))]
        afterTrial_spikes = spike_times[
            (spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time + afterTrial_FR_start) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + afterTrial_FR_end))]

        if previous_hitOrMiss is not None:
            previous_trial_spikes = spike_times[
                (spike_times > previous_hitOrMiss['Trial_onset'] + breakpoint_offset_time) &
                (spike_times < (previous_hitOrMiss['Trial_onset'] + breakpoint_offset_time + stim_duration_for_fr_s))]

            previous_afterTrial_spikes = spike_times[
                (spike_times > previous_hitOrMiss['Trial_onset'] + breakpoint_offset_time + afterTrial_FR_start) &
                (spike_times < (previous_hitOrMiss['Trial_onset'] + breakpoint_offset_time + afterTrial_FR_end))]

            if previous_previous_cr is not None:
                previous_nonAM_spikes = spike_times[
                (spike_times > previous_previous_cr['Trial_onset'] + breakpoint_offset_time) &
                (spike_times < (previous_previous_cr['Trial_onset'] + breakpoint_offset_time + baseline_duration_for_fr_s))]
            else:
                previous_nonAM_spikes = []

        else:
            previous_trial_spikes = []
            previous_afterTrial_spikes = []
            previous_nonAM_spikes = []


        # Get spikes in the interval [trial_onset - pre_stim_raster; trial_onset + post_stim_raster]
        spikes_around_trial = spike_times[
            (spike_times >= cur_trial['Trial_onset'] + breakpoint_offset_time - pre_stim_raster) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + post_stim_raster))]

        # Zero center around trial onset
        spikes_around_trial -= (cur_trial['Trial_onset'] + breakpoint_offset_time)

        timestamped_trial_spikes.append(spikes_around_trial)

        # FR calculations
        cur_nonAM_FR = len(nonAM_spikes) / baseline_duration_for_fr_s
        cur_trial_FR = len(trial_spikes) / stim_duration_for_fr_s
        cur_afterTrial_FR = len(afterTrial_spikes) / (afterTrial_FR_end - afterTrial_FR_start)
        previous_nonAM_FR = len(previous_nonAM_spikes) / baseline_duration_for_fr_s
        previous_trial_FR = len(previous_trial_spikes) / stim_duration_for_fr_s
        previous_afterTrial_FR = len(previous_afterTrial_spikes) / (afterTrial_FR_end - afterTrial_FR_start)
        nonAM_FR_list.append(cur_nonAM_FR)
        trial_FR_list.append(cur_trial_FR)
        afterTrial_FR_list.append(cur_afterTrial_FR)
        previous_nonAM_FR_list.append(previous_nonAM_FR)
        previous_trial_FR_list.append(previous_trial_FR)
        previous_afterTrial_FR_list.append(previous_afterTrial_FR)

        # Spike counts for eventual binomial glms
        nonAM_spikeCount_list.append(len(nonAM_spikes))
        trial_spikeCount_list.append(len(trial_spikes))
        afterTrial_spikeCount_list.append(len(afterTrial_spikes))
        previous_nonAM_spikeCount_list.append(len(previous_nonAM_spikes))
        previous_trial_spikeCount_list.append(len(previous_trial_spikes))
        previous_afterTrial_spikeCount_list.append(len(previous_afterTrial_spikes))

    if csv_pre_name is None:
        csv_name = ''
    else:
        csv_name = csv_pre_name + '_AMsound_firing_rate.csv'

    # write or append
    if first_cell_flag:
        write_or_append_flag = 'w'
    else:
        write_or_append_flag = 'a'
    with open(output_path + sep + csv_name, write_or_append_flag, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Write header if first cell
        if write_or_append_flag == 'w':
            writer.writerow(['Unit'] + ['Key_file'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] + ['ShockFlag'] +
                            ['Hit'] + ['Miss'] + ['CR'] + ['FA'] + ['Period'] +
                            ['FR_Hz'] + ['Spike_count'] + ['Spout_onsets_Hz'] + ['Spout_offsets_Hz']
                            )
        for dummy_idx in range(0, len(nonAM_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]
            for (trial_period, FR_list, spikeCount_list, spoutOn_frequency_list, spoutOff_frequency_list) in \
                zip(('Baseline', 'Trial', 'Aftertrial',
                     'previous_Baseline', 'previous_Trial', 'previous_Aftertrial'),
                    (nonAM_FR_list, trial_FR_list, afterTrial_FR_list,
                     previous_nonAM_FR_list, previous_trial_FR_list, previous_afterTrial_FR_list),
                    (nonAM_spikeCount_list, trial_spikeCount_list, afterTrial_spikeCount_list,
                     previous_nonAM_spikeCount_list, previous_trial_spikeCount_list, previous_afterTrial_spikeCount_list),
                    (nonAM_spoutOn_frequency_list, trial_spoutOn_frequency_list, postStim_spoutOn_frequency_list,
                     np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list))),
                    (nonAM_spoutOff_frequency_list, trial_spoutOff_frequency_list, postStim_spoutOff_frequency_list,
                     np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)), np.repeat(np.NaN, len(nonAM_FR_list)))):

                writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                                [cur_row['ShockFlag']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                [trial_period] + [FR_list[dummy_idx]] +
                                [spikeCount_list[dummy_idx]] +
                                [spoutOn_frequency_list[dummy_idx]] +
                                [spoutOff_frequency_list[dummy_idx]])

            # writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
            #                 [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
            #                 [cur_row['ShockFlag']] +
            #                 [cur_row['Hit']] + [cur_row['Miss']] +
            #                 [cur_row['CR']] + [cur_row['FA']] +
            #                 ['Baseline'] + [nonAM_FR_list[dummy_idx]] +
            #                 [nonAM_spikeCount_list[dummy_idx]] +
            #                 [nonAM_spoutOn_frequency_list[dummy_idx]] +
            #                 [nonAM_spoutOff_frequency_list[dummy_idx]])
            #
            # writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
            #                 [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
            #                 [cur_row['ShockFlag']] +
            #                 [cur_row['Hit']] + [cur_row['Miss']] +
            #                 [cur_row['CR']] + [cur_row['FA']] +
            #                 ['Trial'] + [trial_FR_list[dummy_idx]] +
            #                 [trial_spikeCount_list[dummy_idx]] +
            #                 [trial_spoutOn_frequency_list[dummy_idx]] +
            #                 [trial_spoutOff_frequency_list[dummy_idx]])
            #
            # writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
            #                 [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
            #                 [cur_row['ShockFlag']] +
            #                 [cur_row['Hit']] + [cur_row['Miss']] +
            #                 [cur_row['CR']] + [cur_row['FA']] +
            #                 ['Aftertrial'] + [afterTrial_FR_list[dummy_idx]] +
            #                 [afterTrial_spikeCount_list[dummy_idx]] +
            #                 [postStim_spoutOn_frequency_list[dummy_idx]] +
            #                 [postStim_spoutOff_frequency_list[dummy_idx]])

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'Hit', 'Miss', 'CR', 'FA'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = np.round(relevant_key_times['AMdepth'].values, 2)

    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = timestamped_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Baseline_spikeCount'] = nonAM_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spikeCount'] = trial_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = trial_FR_list
    cur_unitData["Session"][trialInfo_filename]['Previous_Baseline_spikeCount'] = previous_nonAM_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Previous_Baseline_FR'] = previous_nonAM_FR_list
    cur_unitData["Session"][trialInfo_filename]['Previous_Trial_spikeCount'] = previous_trial_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Previous_Trial_FR'] = previous_trial_FR_list
    # cur_unitData["Session"][trialInfo_filename]['All_spikes'] = spike_times - breakpoint_offset_time  # Subtract offset

    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOn_frequency'] = nonAM_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOn_frequency'] = trial_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOff_frequency'] = nonAM_spoutOff_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOff_frequency'] = trial_spoutOff_frequency_list

    cur_unitData["Session"][trialInfo_filename]['SpoutOn_times_during_trial'] = spoutOn_timestamps_list
    cur_unitData["Session"][trialInfo_filename]['SpoutOff_times_during_trial'] = spoutOff_timestamps_list

    return cur_unitData


def get_trial_info_only(key_path_info,
                        cur_unitData):
    """
    Get info about trials (trialType, shockFlag, AMdepth etc, without computing firing rates)
    """
    # Load key files
    info_key_times = read_csv(key_path_info)

    # Grab GO trials and FA trials for stim response then walk back to get the immediately preceding CR trial
    # One complicating factor is spout onset/offset but I'll ignore this for now
    # Also ignore reminder trials
    relevant_key_times = info_key_times[
        ((info_key_times['TrialType'] == 0) | (info_key_times['FA'] == 1)) & (info_key_times['Reminder'] == 0)]

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'ShockFlag', 'Hit', 'Miss', 'CR', 'FA'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = np.round(relevant_key_times['AMdepth'].values, 2)

    return cur_unitData
