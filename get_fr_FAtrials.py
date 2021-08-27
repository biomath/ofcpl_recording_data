from intrastim_correlation import *
from pandas import read_csv


def get_fr_FAtrials(memory_name,
                   key_path_info,
                   unit_name,
                   output_path,
                   sampling_rate=24414.0625,
                   csv_pre_name=None,
                   first_cell_flag=True,
                   breakpoint_offset=0,
                   baseline_duration_for_fr_s=1,
                   stim_duration_for_fr_s=1):

    # Load key files
    info_key_times = read_csv(key_path_info)

    spike_times = np.genfromtxt(memory_name)

    # The default is 0, so this will only make a difference if set at function call
    breakpoint_offset_time = breakpoint_offset / sampling_rate

    # Grab GO trials for stim response then walk back to get the immediately preceding NO-GO
    # One complicating factor is spout onset/offset but I'll ignore this for now
    # Also ignore reminder trials
    relevant_key_times = info_key_times[(info_key_times['FA'] == 1) & (info_key_times['Reminder'] == 0)]

    # Now grab spike times
    # Baseline will be the correct rejection trial immediately preceding a false alarm trial
    baseline_stim_FR_list = list()
    baseline_stim_spikeCount_list = list()
    for _, cur_trial in relevant_key_times.iterrows():
        # Get spike times around the current stimulus onset
        # For baseline, go to the previous trial (NO-GO)
        previous_trial = info_key_times[(info_key_times['CR'] == 1) &
                                        (info_key_times['TrialID'] < cur_trial['TrialID'])].iloc[-1]
        baseline_spikes = spike_times[
            (previous_trial['Trial_onset'].values[0] + breakpoint_offset_time < spike_times) &
            (spike_times < previous_trial['Trial_onset'].values[0] + breakpoint_offset_time + baseline_duration_for_fr_s)]

        stimulus_spikes = spike_times[(spike_times > cur_trial['Trial_onset'] + breakpoint_offset_time) &
                                      (spike_times < (cur_trial['Trial_onset'] + breakpoint_offset_time + stim_duration_for_fr_s))]

        # FR calculations
        cur_baseline_FR = len(baseline_spikes) / baseline_duration_for_fr_s
        cur_stim_FR = len(stimulus_spikes) / stim_duration_for_fr_s
        baseline_stim_FR_list.append((cur_baseline_FR, cur_stim_FR))

        # Spike counts for eventual binomial glms
        baseline_stim_spikeCount_list.append((len(baseline_spikes), len(stimulus_spikes)))

    if csv_pre_name is None:
        pass
    else:
        csv_name = csv_pre_name + '_FAtrials_firing_rate.csv'

        # write or append
        if first_cell_flag:
            write_or_append_flag = 'w'
        else:
            write_or_append_flag = 'a'
        with open(output_path + '\\' + csv_name, write_or_append_flag, newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write header if first cell
            if write_or_append_flag == 'w':
                writer.writerow(['Unit'] + ['Key_file'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] +
                                ['Hit'] + ['Miss'] + ['CR'] + ['FA'] + ['Period'] +
                                ['FR_Hz'] + ['Spike_count'])
            for dummy_idx in range(0, len(baseline_stim_FR_list)):
                cur_row = relevant_key_times.iloc[dummy_idx, :]
                writer.writerow([unit_name] + [split("\\\\", key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                ['Baseline'] + [baseline_stim_FR_list[dummy_idx][0]] +
                                [baseline_stim_spikeCount_list[dummy_idx][0]])

                writer.writerow([unit_name] + [split("\\\\", key_path_info)[-1][:-4]] +
                                [cur_row['TrialID']] + [round(cur_row['AMdepth'], 2)] + [cur_row['Reminder']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                ['Trial'] + [baseline_stim_FR_list[dummy_idx][1]] +
                                [baseline_stim_spikeCount_list[dummy_idx][1]])
