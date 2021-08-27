from intrastim_correlation import *
from pandas import read_csv
from os.path import sep
import platform
import json

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
                    csv_pre_name=None,
                    first_cell_flag=True,
                    breakpoint_offset=0,
                    baseline_duration_for_fr_s=0.5,
                    stim_duration_for_fr_s=0.5,
                    pre_stim_raster=2.,  # For timestamped spikeTimes
                    post_stim_raster=4.,  # For timestamped spikeTimes
                    cur_unitData=None):
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
    baseline_FR_list = list()
    stim_FR_list = list()
    baseline_spikeCount_list = list()
    stim_spikeCount_list = list()
    baseline_spoutOn_frequency_list = list()
    stim_spoutOn_frequency_list = list()
    baseline_spoutOff_frequency_list = list()
    stim_spoutOff_frequency_list = list()
    spoutOn_timestamps_list = list()
    spoutOff_timestamps_list = list()
    timestamped_trial_spikes = list()
    for _, cur_trial in relevant_key_times.iterrows():
        # Get spike times around the current stimulus onset
        # For baseline, go to the previous trial that resulted in a correct rejection (NO-GO) which
        # may not be the immediately preceding trial
        try:
            previous_trial = info_key_times[(info_key_times['CR'] == 1) &
                                            (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
        except IndexError:  # In case there is no CR before a hit, skip trial
            continue

        # Count the number of spout onsets and offsets during the current trials; could be interesting...
        # Also transform to Hz in case I decide to change the window in the future
        baseline_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 previous_trial['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 previous_trial['Trial_onset'] + baseline_duration_for_fr_s)]['Spout_onset'].values
        baseline_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 previous_trial['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 previous_trial['Trial_onset'] + baseline_duration_for_fr_s)]['Spout_offset'].values

        stim_spoutOn = \
            spout_key_times[
                (spout_key_times['Spout_onset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_onset'].values <
                 cur_trial['Trial_onset'] + stim_duration_for_fr_s)]['Spout_onset'].values
        stim_spoutOff = \
            spout_key_times[
                (spout_key_times['Spout_offset'].values >
                 cur_trial['Trial_onset']) &
                (spout_key_times['Spout_offset'].values <
                 cur_trial['Trial_onset'] + stim_duration_for_fr_s)]['Spout_offset'].values

        # DEBUGGING:
        # if np.sum([baseline_spoutOn, baseline_spoutOff, stim_spoutOn, stim_spoutOff]) > 1:
        #     print()

        # Append to lists
        baseline_spoutOn_frequency_list.append(len(baseline_spoutOn) / baseline_duration_for_fr_s)
        stim_spoutOn_frequency_list.append(len(stim_spoutOn) / stim_duration_for_fr_s)
        baseline_spoutOff_frequency_list.append(len(baseline_spoutOff) / baseline_duration_for_fr_s)
        stim_spoutOff_frequency_list.append(len(stim_spoutOff) / stim_duration_for_fr_s)

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
        baseline_spikes = spike_times[
            (previous_trial['Trial_onset'] + breakpoint_offset_time < spike_times) &
            (spike_times < previous_trial['Trial_onset'] + breakpoint_offset_time + baseline_duration_for_fr_s)]

        stim_spikes = spike_times[(spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time) &
                                      (spike_times < (cur_trial[
                                                          'Trial_onset'] + breakpoint_offset_time + stim_duration_for_fr_s))]

        # Get spikes in the interval [trial_onset - pre_stim_raster; trial_onset + post_stim_raster]
        spikes_around_trial = spike_times[
            (spike_times >= cur_trial['Trial_onset'] + breakpoint_offset_time - pre_stim_raster) &
            (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + post_stim_raster))]

        # Zero center around trial onset
        spikes_around_trial -= (cur_trial['Trial_onset'] + breakpoint_offset_time)

        timestamped_trial_spikes.append(spikes_around_trial)

        # FR calculations
        cur_baseline_FR = len(baseline_spikes) / baseline_duration_for_fr_s
        cur_stim_FR = len(stim_spikes) / stim_duration_for_fr_s
        baseline_FR_list.append(cur_baseline_FR)
        stim_FR_list.append(cur_stim_FR)

        # Spike counts for eventual binomial glms
        baseline_spikeCount_list.append(len(baseline_spikes))
        stim_spikeCount_list.append(len(stim_spikes))

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
            writer.writerow(['Unit'] + ['Key_file'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] +
                            ['Hit'] + ['Miss'] + ['CR'] + ['FA'] + ['Period'] +
                            ['FR_Hz'] + ['Spike_count'] + ['Spout_onsets_Hz'] + ['Spout_offsets_Hz'])
        for dummy_idx in range(0, len(baseline_FR_list)):
            cur_row = relevant_key_times.iloc[dummy_idx, :]
            writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                            [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                            [cur_row['Hit']] + [cur_row['Miss']] +
                            [cur_row['CR']] + [cur_row['FA']] +
                            ['Baseline'] + [baseline_FR_list[dummy_idx]]+
                            [baseline_spikeCount_list[dummy_idx]] +
                            [baseline_spoutOn_frequency_list[dummy_idx]] +
                            [baseline_spoutOff_frequency_list[dummy_idx]])

            writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
                            [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                            [cur_row['Hit']] + [cur_row['Miss']] +
                            [cur_row['CR']] + [cur_row['FA']] +
                            ['Trial'] + [stim_FR_list[dummy_idx]]+
                            [stim_spikeCount_list[dummy_idx]] +
                            [stim_spoutOn_frequency_list[dummy_idx]] +
                            [stim_spoutOff_frequency_list[dummy_idx]])

    # Add all info to unitData
    trialInfo_filename = split(REGEX_SEP, key_path_info)[-1][:-4]
    for key_name in ('TrialID', 'Reminder', 'Hit', 'Miss', 'CR', 'FA'):
        cur_unitData["Session"][trialInfo_filename][key_name] = relevant_key_times[key_name].values

    cur_unitData["Session"][trialInfo_filename]['AMdepth'] = np.round(relevant_key_times['AMdepth'].values, 2)

    cur_unitData["Session"][trialInfo_filename]['Trial_spikes'] = timestamped_trial_spikes
    cur_unitData["Session"][trialInfo_filename]['Baseline_spikeCount'] = baseline_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_FR'] = baseline_FR_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spikeCount'] = stim_spikeCount_list
    cur_unitData["Session"][trialInfo_filename]['Trial_FR'] = stim_FR_list
    # cur_unitData["Session"][trialInfo_filename]['All_spikes'] = spike_times - breakpoint_offset_time  # Subtract offset

    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOn_frequency'] = baseline_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOn_frequency'] = stim_spoutOn_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Baseline_spoutOff_frequency'] = baseline_spoutOff_frequency_list
    cur_unitData["Session"][trialInfo_filename]['Trial_spoutOff_frequency'] = stim_spoutOff_frequency_list

    cur_unitData["Session"][trialInfo_filename]['SpoutOn_times_during_trial'] = spoutOn_timestamps_list
    cur_unitData["Session"][trialInfo_filename]['SpoutOff_times_during_trial'] = spoutOff_timestamps_list

    return cur_unitData

    #
    # with open(output_path + sep + csv_name, write_or_append_flag, newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     # Write header if first cell
    #     if write_or_append_flag == 'w':
    #         writer.writerow(['Unit'] + ['Key_file'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] +
    #                         ['Hit'] + ['Miss'] + ['CR'] + ['FA'] +
    #                         ['Timestamped_spikes'])
    #     for dummy_idx in range(0, len(timestamped_trial_spikes)):
    #         cur_row = relevant_key_times.iloc[dummy_idx, :]
    #         writer.writerow([unit_name] + [split(REGEX_SEP, key_path_info)[-1][:-4]] +
    #                         [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
    #                         [cur_row['Hit']] + [cur_row['Miss']] +
    #                         [cur_row['CR']] + [cur_row['FA']] +
    #                         [timestamped_trial_spikes[dummy_idx]])
