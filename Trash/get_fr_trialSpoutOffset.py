from intrastim_correlation import *
from pandas import read_csv
from os.path import sep
import platform

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep*2
else:
    REGEX_SEP = sep

def get_fr_spoutOffset(memory_name,
                       key_path_spout,
                       key_path_info,
                       unit_name,
                       output_path,
                       csv_pre_name=None,
                       first_cell_flag=True,
                       baseline_duration_for_fr_s=0.5,
                       stim_duration_for_fr_s=0.5):

    # Load key files
    spout_key_times = read_csv(key_path_spout)
    info_key_times = read_csv(key_path_info)

    spike_times = np.genfromtxt(memory_name)

    # Grab the last spout offset in a trial (the one triggering the hit or FA)
    relevant_key_times = info_key_times[(info_key_times['Hit'] == 1) | (info_key_times['FA'] == 1) | (info_key_times['Miss'] == 1)]
    # Now grab spike times
    baseline_stim_FR_list = list()
    baseline_stim_spikeCount_list = list()
    for _, cur_timestamp in relevant_key_times.iterrows():
        # If there's more than one offset value, grab the last
        try:
            if cur_timestamp['Miss'] == 1:
                # If there's a shock, grab the first spout_offset within a 0.5 s window of the shock
                curr_offset = spout_key_times[
                    (spout_key_times['Spout_offset'] > cur_timestamp['Trial_offset']) &
                    (spout_key_times['Spout_offset'] < cur_timestamp['Trial_offset'] + 0.5)]['Spout_offset'].values[0]
            else:
                curr_offset = spout_key_times[
                    (spout_key_times['Spout_offset'] < cur_timestamp['Trial_offset']) &
                    (spout_key_times['Spout_offset'] > cur_timestamp['Trial_onset'])]['Spout_offset'].values[-1]

        # Rarely there are trials where the animal leaves the trial immediately after an onset, and the timestamps
        # will be out of sync causing the offset time to be non-existant. Skip these trials for now
        except IndexError:
            continue

        # Get spike times around the current stimulus onset
        baseline_spikes = spike_times[((curr_offset - baseline_duration_for_fr_s) < spike_times) &
                                      (spike_times < curr_offset)]

        # Avoid the shock artifact by sliding up 0.5 ms
        # if cur_timestamp['Miss'] == 1:
        stimulus_spikes = spike_times[(spike_times > curr_offset + 0.5) &
                                          (spike_times < (curr_offset + 0.5 + stim_duration_for_fr_s))]
        # else:
        #     stimulus_spikes = spike_times[(spike_times > curr_offset) &
        #                                   (spike_times < (curr_offset + stim_duration_for_fr_s))]

        # FR calculations
        cur_baseline_FR = len(baseline_spikes) / baseline_duration_for_fr_s
        cur_stim_FR = len(stimulus_spikes) / stim_duration_for_fr_s
        baseline_stim_FR_list.append((cur_baseline_FR, cur_stim_FR))

        # Spike counts for eventual binomial glms
        baseline_stim_spikeCount_list.append((len(baseline_spikes), len(stimulus_spikes)))

    if csv_pre_name is None:
        pass
    else:
        csv_name = csv_pre_name + '_spoutOffset_firing_rate.csv'

        # write or append
        if first_cell_flag:
            wa_flag = 'w'
        else:
            wa_flag = 'a'
        with open(output_path + sep + csv_name, wa_flag, newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write header if first cell
            if wa_flag == 'w':
                writer.writerow(['Unit'] + ['TrialID'] + ['AMdepth'] + ['Reminder'] +
                                ['Hit'] + ['Miss'] + ['CR'] + ['FA'] + ['Period'] +
                                ['FR_Hz'] + ['Spike_count'])
            for dummy_idx in range(0, len(baseline_stim_FR_list)):
                cur_row = relevant_key_times.iloc[dummy_idx, :]
                writer.writerow([unit_name] + [cur_row['TrialID']] + [cur_row['AMdepth']] + [cur_row['Reminder']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                ['Baseline'] + [baseline_stim_FR_list[dummy_idx][0]] +
                                [baseline_stim_spikeCount_list[dummy_idx][0]])

                writer.writerow([unit_name] + [cur_row['TrialID']] + [cur_row['AMdepth']] + [cur_row['Reminder']] +
                                [cur_row['Hit']] + [cur_row['Miss']] +
                                [cur_row['CR']] + [cur_row['FA']] +
                                ['Stimulus'] + [baseline_stim_FR_list[dummy_idx][1]] +
                                [baseline_stim_spikeCount_list[dummy_idx][1]])
