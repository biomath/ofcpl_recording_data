from os.path import sep
from pandas import read_csv
from intrastim_correlation import *

def get_fr_allSpoutTimestamps(memory_name,
                              key_path_spout,
                              unit_name,
                              output_path,
                              csv_pre_name=None,
                              first_cell_flag=True,
                              baseline_duration_for_fr_s=0.5,
                              stim_duration_for_fr_s=0.5,
                              pre_stimulus_raster=2.,  # For timestamped spikeTimes
                              post_stimulus_raster=4.,  # For timestamped spikeTimes
                              breakpoint_offset=0,
                              session_name=None,
                              cur_unitData=None):

    # Load key files
    spout_key_times = read_csv(key_path_spout)

    spike_times = np.genfromtxt(memory_name)

    # The default is 0, so this will only make a difference if set at function call
    # breakpoint_offset_time = breakpoint_offset / sampling_rate # OBSOLETE: offsets are now in seconds
    breakpoint_offset_time = breakpoint_offset

    offset_baseline_stim_FR_list = list()
    offset_baseline_stim_spikeCount_list = list()
    offset_raster_list = list()
    onset_baseline_stim_FR_list = list()
    onset_baseline_stim_spikeCount_list = list()
    onset_raster_list = list()
    for _, cur_timestamp in spout_key_times.iterrows():
        offset_timestamp = cur_timestamp['Spout_offset'] + breakpoint_offset_time
        onset_timestamp = cur_timestamp['Spout_onset'] + breakpoint_offset_time

        # Skip brief spout events; <2 s offset - onset;
        if offset_timestamp - onset_timestamp < 2:
            continue

        # Get spike times around the current spout offset
        offset_baseline_spikes = spike_times[((offset_timestamp - baseline_duration_for_fr_s) < spike_times) &
                                      (spike_times < offset_timestamp)]
        onset_baseline_spikes = spike_times[((onset_timestamp - baseline_duration_for_fr_s) < spike_times) &
                                      (spike_times < onset_timestamp)]

        offset_response_spikes = spike_times[(spike_times > offset_timestamp) &
                                          (spike_times < (offset_timestamp + stim_duration_for_fr_s))]

        onset_response_spikes = spike_times[(spike_times > onset_timestamp) &
                                          (spike_times < (onset_timestamp + stim_duration_for_fr_s))]

        # Get spikes in the interval [timestamp - pre_stim_raster; timestamp + post_stim_raster]
        offset_raster = spike_times[((offset_timestamp - pre_stimulus_raster) < spike_times) &
                                    (spike_times < (offset_timestamp + post_stimulus_raster))]
        onset_raster = spike_times[((onset_timestamp - pre_stimulus_raster) < spike_times) &
                                   (spike_times < (onset_timestamp + post_stimulus_raster))]

        # Zero center around trial onset
        offset_raster -= offset_timestamp
        onset_raster -= onset_timestamp

        offset_raster_list.append(offset_raster)
        onset_raster_list.append(onset_raster)

        # FR calculations
        offset_cur_baseline_FR = len(offset_baseline_spikes) / baseline_duration_for_fr_s
        offset_cur_stim_FR = len(offset_response_spikes) / stim_duration_for_fr_s
        offset_baseline_stim_FR_list.append((offset_cur_baseline_FR, offset_cur_stim_FR))

        onset_cur_baseline_FR = len(onset_baseline_spikes) / baseline_duration_for_fr_s
        onset_cur_stim_FR = len(onset_response_spikes) / stim_duration_for_fr_s
        onset_baseline_stim_FR_list.append((onset_cur_baseline_FR, onset_cur_stim_FR))

        # Spike counts for eventual binomial glms
        offset_baseline_stim_spikeCount_list.append((len(offset_baseline_spikes), len(offset_response_spikes)))
        onset_baseline_stim_spikeCount_list.append((len(onset_baseline_spikes), len(onset_response_spikes)))


    if csv_pre_name is None:
        pass
    else:
        csv_name = csv_pre_name + '_allSpoutOnsetOffset_firing_rate.csv'

        # write or append
        if first_cell_flag:
            wa_flag = 'w'
        else:
            wa_flag = 'a'
        with open(output_path + sep + csv_name, wa_flag, newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write header if first cell
            if wa_flag == 'w':
                writer.writerow(['Unit'] + ['Timestamp'] + ['Period'] + ['Onset_or_offset'] +
                                ['FR_Hz'] + ['Spike_count'])
            for dummy_idx in range(0, len(offset_baseline_stim_FR_list)):
                cur_row = spout_key_times.iloc[dummy_idx, :]
                writer.writerow([unit_name] + [cur_row['Spout_offset']] + ['Baseline'] + ['Offset'] +
                                [offset_baseline_stim_FR_list[dummy_idx][0]] +
                                [offset_baseline_stim_spikeCount_list[dummy_idx][0]])

                writer.writerow([unit_name] + [cur_row['Spout_offset']] + ['Spout'] + ['Offset'] +
                                [offset_baseline_stim_FR_list[dummy_idx][1]] +
                                [offset_baseline_stim_spikeCount_list[dummy_idx][1]])

                writer.writerow([unit_name] + [cur_row['Spout_onset']] + ['Baseline'] + ['Onset'] +
                                [onset_baseline_stim_FR_list[dummy_idx][0]] +
                                [onset_baseline_stim_spikeCount_list[dummy_idx][0]])

                writer.writerow([unit_name] + [cur_row['Spout_onset']] + ['Spout'] + ['Onset'] +
                                [onset_baseline_stim_FR_list[dummy_idx][1]] +
                                [onset_baseline_stim_spikeCount_list[dummy_idx][1]])



    # Add all info to unitData
    # Offset
    cur_unitData["Session"][session_name]['Offset_timestamps'] = spout_key_times['Spout_offset'].values
    cur_unitData["Session"][session_name]['Offset_baseline_FR'] = [x[0] for x in offset_baseline_stim_FR_list]
    cur_unitData["Session"][session_name]['Offset_trial_FR'] = [x[1] for x in offset_baseline_stim_FR_list]
    cur_unitData["Session"][session_name]['Offset_baseline_spikeCount'] = [x[0] for x in offset_baseline_stim_spikeCount_list]
    cur_unitData["Session"][session_name]['Offset_trial_spikeCount'] = [x[1] for x in offset_baseline_stim_spikeCount_list]
    cur_unitData["Session"][session_name]['Offset_rasters'] = offset_raster_list
    # Onset
    cur_unitData["Session"][session_name]['Onset_timestamps'] = spout_key_times['Spout_onset'].values
    cur_unitData["Session"][session_name]['Onset_baseline_FR'] = [x[0] for x in onset_baseline_stim_FR_list]
    cur_unitData["Session"][session_name]['Onset_trial_FR'] = [x[1] for x in onset_baseline_stim_FR_list]
    cur_unitData["Session"][session_name]['Onset_baseline_spikeCount'] = [x[0] for x in onset_baseline_stim_spikeCount_list]
    cur_unitData["Session"][session_name]['Onset_trial_spikeCount'] = [x[1] for x in onset_baseline_stim_spikeCount_list]
    cur_unitData["Session"][session_name]['Onset_rasters'] = onset_raster_list

    return cur_unitData