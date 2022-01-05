from intrastim_correlation import *
import matplotlib as mpl
from pandas import read_csv, DataFrame
from os import makedirs
from os.path import sep
from re import split
from glob import glob
from datetime import datetime
from format_axes import *
from matplotlib.backends.backend_pdf import PdfPages
import platform

import warnings
import multiprocessing as mp
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep*2
else:
    REGEX_SEP = sep

def gaussian_moving_average_fft(points, sigma):
    return convolve_fft(points, Gaussian1DKernel(stddev=sigma))


def gaussian_smoothing(hist, edges, pre_stimulus_baseline, window):
    """
    First, center PSTH around 0 using an average of baseline then smooth with gaussian window.
    Finally, normalize PSTH so that the peak or valley assumes a value of 1 or -1
    :param hist:
    :param window_ms:
    :return:
    """

    # Center PSTH
    baseline_points_mask = (edges > -pre_stimulus_baseline) & (edges < 0)
    ret_psth = hist - np.mean(hist[baseline_points_mask[:-1]])

    # Smooth
    ret_psth = gaussian_moving_average_fft(ret_psth, window)

    # Rescale
    abs_peak = np.max(np.abs(ret_psth))
    ret_psth = ret_psth / abs_peak

    return ret_psth



def common_psth_engine(spike_times,
                       key_times,
                       pre_stimulus_raster, post_stimulus_raster,
                       pre_stimulus_baseline=None, gaussian_window=None,
                       ax_raster=None, ax_psth=None, ax_gaussian=None,
                       breakpoint_offset=None,
                       hist_bin_size_ms=10,
                       do_plot=True,
                       rasterize=True):
    number_of_stimulus_repetitions = len(key_times)

    bin_size = hist_bin_size_ms / 1000  # in s
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster+bin_size, bin_size)

    # Loop through each stimulus presentation
    raster_trial_counter = number_of_stimulus_repetitions
    # raster_trial_counter = 0
    relative_times = list()
    for cur_stim_time in key_times:
        # offset_stim_time = cur_stim_time + breakpoint_offset / sampling_rate
        offset_stim_time = cur_stim_time + breakpoint_offset  # breakpoints are in seconds now

        # Get spike times around the current stimulus onset
        times_to_plot = spike_times[((offset_stim_time - pre_stimulus_raster) < spike_times) &
                                    (spike_times < (offset_stim_time + post_stimulus_raster))]

        # Zero-center spike times
        curr_relative_times = times_to_plot - offset_stim_time
        if do_plot:
            ax_raster.plot(curr_relative_times,
                           np.repeat(raster_trial_counter, len(curr_relative_times)),
                           'k|',
                           rasterized=rasterize)
            raster_trial_counter -= 1

        relative_times.append([x for x in curr_relative_times])

    relative_times = [item for sublist in relative_times for item in sublist]

    # Only to get y labels. Later, use this to plot as well
    hist, edges = np.histogram(relative_times, bins=bin_cuts)

    # Change hist to spike rate before appending
    hist = np.round(hist / number_of_stimulus_repetitions / bin_size, 2)

    # Gaussian smoothing for Gaussian PSTH
    gaussian_psth = None
    if gaussian_window is not None:
        gaussian_psth = gaussian_smoothing(hist, edges, pre_stimulus_baseline, gaussian_window)

    if do_plot:
        ax_psth.bar(bin_cuts[:-1], hist, 0.01, color='k', align='edge')
        # ax_psth.hist(relative_times, bins=bin_cuts, edgecolor=None, facecolor='k')

        if gaussian_window is not None:
            gaussian_psth_x = np.linspace(-pre_stimulus_raster, post_stimulus_raster, len(bin_cuts[:-1]))
            ax_gaussian.plot(gaussian_psth_x, gaussian_psth, color='steelblue')

    return hist, gaussian_psth


def plot_psth_spoutOffset(memory_paths,
                          all_key_path,
                          pre_stimulus_raster, post_stimulus_raster,
                          recording_type,  # synapse or intan
                          hist_bin_size_ms=10,
                          pre_stimulus_baseline=0.5,
                          gaussian_window=5,
                          cur_breakpoint_df=DataFrame(),
                          uniform_psth_y_lim=False,
                          raster_max_y_lim=500.5):
    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = sep.join([OUTPUT_PATH, subject_id, cur_date, "Spout offset"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys

    key_paths = glob(all_key_path + sep + subject_id + '*' +
                     cur_date + "*_spoutTimestamps.csv")

    if len(key_paths) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths = glob(all_key_path + sep + subject_id + '*' +
                         cur_date + "*_spoutTimestamps.csv")

    # make sure they're ordered by session time for plotting
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + sep + output_name + '_spoutOffset.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            if "Aversive" not in key_path and "Active" not in key_path:  # skip passive stuff
                continue
            temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
            if recording_type == 'synapse':
                temp_session = temp_session[1]
            elif recording_type == 'intan':
                temp_session = "_".join(temp_session[1:4])
            else:
                print('Recording type not specified. Skipping;...')
                return
            breakpoint_offset_idx = cur_breakpoint_df.index[
                cur_breakpoint_df['Session_file'].str.contains(temp_session)]
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
            except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
                try:
                    breakpoint_offset = cur_breakpoint_df.loc[
                        breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
                except KeyError:
                    breakpoint_offset = 0  # first file; no breakpoint offset needed
                if recording_type == 'synapse':
                    breakpoint_offset = breakpoint_offset / 24414.0625
                elif recording_type == 'intan':
                    breakpoint_offset = breakpoint_offset / 30000
                else:
                    print('Recording type not specified. Skipping;...')
                    return

            # Load key file
            key_times = read_csv(key_path)

            # Set up figure and axs
            plt.clf()
            f = plt.figure()
            ax_psth = f.add_subplot(212)
            ax_raster = f.add_subplot(211, sharex=ax_psth)
            ax_gaussian = ax_psth.twinx()
            # Populate axs
            common_psth_engine(spike_times=spike_times,
                               key_times=key_times['Spout_offset'],
                               pre_stimulus_raster=pre_stimulus_raster,
                               post_stimulus_raster=post_stimulus_raster,
                               ax_psth=ax_psth, ax_raster=ax_raster,
                               pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                               gaussian_window=gaussian_window,
                               hist_bin_size_ms=10,
                               breakpoint_offset=breakpoint_offset,
                               do_plot=True)

            # Format axs
            format_ax(ax_raster)
            format_ax(ax_psth)

            ax_raster.axis('off')

            ax_raster.set_ylim([-0.5, raster_max_y_lim])
            ax_gaussian.set_ylim([-1, 1])

            ax_psth.set_ylabel("Spike rate by trial (Hz)")
            ax_psth.set_xlabel("Time (s)")
            ax_gaussian.set_ylabel("Spike rate (normalized; smoothed)")
            f.suptitle("_".join(split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[0:2]) +
                       "_" + split("_*_", split(REGEX_SEP, memory_paths[0])[-1][:-4])[-1] + ":\n" + "Spout offset")

            pdf.savefig()

            # f.savefig(output_path + sep + "_".join(split("_*_", split(REGEX_SEP, key_name)[-1][:-4])[0:2]) +
            #           "_" + split("_*_", split(REGEX_SEP, memory_name)[-1][:-4])[-1] +
            #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

            plt.close()


def plot_psth_spoutOnset(memory_paths,
                         all_key_path,
                         pre_stimulus_raster, post_stimulus_raster,
                         recording_type,  # synapse or intan
                         hist_bin_size_ms=10,
                         pre_stimulus_baseline=0.5,
                         gaussian_window=5,
                         cur_breakpoint_df=None,
                         uniform_psth_y_lim=False,
                         raster_max_y_lim=500.5):
    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = sep.join([OUTPUT_PATH, subject_id, cur_date, "Spout onset"])
    makedirs(output_subfolder, exist_ok=True)
    # Use subj-session date to grab appropriate keys

    key_paths = glob(all_key_path + sep + subject_id + '*' +
                     cur_date + "*_spoutTimestamps.csv")

    if len(key_paths) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths = glob(all_key_path + sep + subject_id + '*' +
                         cur_date + "*_spoutTimestamps.csv")

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + sep + output_name + '_spoutOnset.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            if "Aversive" not in key_path and "Active" not in key_path:  # skip passive stuff
                continue
            temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
            if recording_type == 'synapse':
                temp_session = temp_session[1]
            elif recording_type == 'intan':
                temp_session = "_".join(temp_session[1:4])
            else:
                print('Recording type not specified. Skipping;...')
                return
            breakpoint_offset_idx = cur_breakpoint_df.index[
                cur_breakpoint_df['Session_file'].str.contains(temp_session)]
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
            except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
                try:
                    breakpoint_offset = cur_breakpoint_df.loc[
                        breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
                except KeyError:
                    breakpoint_offset = 0  # first file; no breakpoint offset needed
                if recording_type == 'synapse':
                    breakpoint_offset = breakpoint_offset / 24414.0625
                elif recording_type == 'intan':
                    breakpoint_offset = breakpoint_offset / 30000
                else:
                    print('Recording type not specified. Skipping;...')
                    return

            # Load key file
            key_times = read_csv(key_path)

            # Set up figure and axs
            plt.clf()
            f = plt.figure()
            ax_psth = f.add_subplot(212)
            ax_raster = f.add_subplot(211, sharex=ax_psth)
            ax_gaussian = ax_psth.twinx()

            # Populate axs
            common_psth_engine(spike_times=spike_times,
                               key_times=key_times['Spout_onset'],
                               pre_stimulus_raster=pre_stimulus_raster,
                               post_stimulus_raster=post_stimulus_raster,
                               ax_psth=ax_psth, ax_raster=ax_raster,
                               pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                               gaussian_window=gaussian_window,
                               hist_bin_size_ms=10,
                               breakpoint_offset=breakpoint_offset,
                               do_plot=True)

            # Format axs
            format_ax(ax_raster)
            format_ax(ax_psth)

            ax_raster.axis('off')

            ax_raster.set_ylim([-0.5, raster_max_y_lim])
            ax_gaussian.set_ylim([-1, 1])
            ax_psth.set_ylabel("Spike rate by trial (Hz)")
            ax_psth.set_xlabel("Time (s)")
            f.suptitle("_".join(split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[0:2]) +
                       "_" + split("_*_", split(REGEX_SEP, memory_paths[0])[-1][:-4])[-1] + ":\n" + "Spout onset")

            pdf.savefig()

            # f.savefig(output_path + sep + "_".join(split("_*_", split(REGEX_SEP, key_name)[-1][:-4])[0:2]) +
            #           "_" + split("_*_", split(REGEX_SEP, memory_name)[-1][:-4])[-1] +
            #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

            plt.close()


def plot_psth_AMdepth(memory_paths,
                      all_key_path,
                      pre_stimulus_raster, post_stimulus_raster,
                      recording_type,  # intan or synapse
                      hist_bin_size_ms=10,
                      pre_stimulus_baseline=0.5,
                      gaussian_window=5,
                      cur_breakpoint_df=None,
                      uniform_psth_y_lim=True,
                      raster_max_y_lim=30.5,
                      override_max_ylim=None):
    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = sep.join([OUTPUT_PATH, subject_id, cur_date, "AM depth"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys

    key_paths = glob(all_key_path + sep + subject_id + '*' +
                     cur_date + "*_trialInfo.csv")

    if len(key_paths) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths = glob(all_key_path + sep + subject_id + '*' +
                         cur_date + "*_trialInfo.csv")

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    if recording_type == 'synapse':
        session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    elif recording_type == 'intan':
        session_times = [split("-*-", split_path[2]) for split_path in split_key_paths]
        session_times = [datetime.strptime("".join(timestamp), '%H%M%S') for timestamp in session_times]
    else:
        print('Recording type not specified. Skipping....')
        return

    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name = output_subfolder + sep + output_name + '_AMdepth'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    # Get max y lim for PSTHs across conditions (e.g. Pre, Active, Post) for the current unit
    if override_max_ylim is None:
        psth_max_ylim = None
        if uniform_psth_y_lim:
            psth_max_ylim = 0
            for key_idx, key_path in enumerate(key_paths):
                temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
                if recording_type == 'synapse':
                    temp_session = temp_session[1]
                elif recording_type == 'intan':
                    temp_session = "_".join(temp_session[1:4])
                else:
                    print('Recording type not specified. Skipping;...')
                    return
                breakpoint_offset_idx = cur_breakpoint_df.index[
                    cur_breakpoint_df['Session_file'].str.contains(temp_session)]
                try:
                    breakpoint_offset = cur_breakpoint_df.loc[
                        breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
                except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
                    try:
                        breakpoint_offset = cur_breakpoint_df.loc[
                            breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
                    except KeyError:
                        breakpoint_offset = 0  # first file; no breakpoint offset needed
                    if recording_type == 'synapse':
                        breakpoint_offset = breakpoint_offset / 24414.0625
                    elif recording_type == 'intan':
                        breakpoint_offset = breakpoint_offset / 30000
                    else:
                        print('Recording type not specified. Skipping;...')
                        return

                # Load key file
                key_times = read_csv(key_path)
                for stim in sorted(list(set(key_times['AMdepth']))):
                    # skip no-gos
                    # if stim == 0:
                    #     continue

                    # Hard code here to eliminate all trials above 75 for SUBJ 174
                    if subject_id == 'SUBJ-ID-174':
                        key_times = key_times[key_times['TrialID'] < 75]

                    cur_key_times = key_times[round(key_times['AMdepth'], 2) == round(stim, 2)]['Trial_onset']

                    ## do something really inefficient here to hold y_axis uniform across all stimuli and treatments
                    hist, _ = common_psth_engine(spike_times,
                                                 cur_key_times,
                                                 pre_stimulus_raster, post_stimulus_raster,
                                                 breakpoint_offset=breakpoint_offset,
                                                 hist_bin_size_ms=hist_bin_size_ms,
                                                 do_plot=False)
                    if np.max(hist) > psth_max_ylim:
                        psth_max_ylim = np.max(hist)
    else:
        psth_max_ylim = override_max_ylim

    if OUTPUT_TYPE == 'pdf':
        pdf = PdfPages(output_name + '.pdf')

    # with PdfPages(output_name_pdf) as pdf:
    for key_idx, key_path in enumerate(key_paths):
        temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
        if recording_type == 'synapse':
            temp_session = temp_session[1]
        elif recording_type == 'intan':
            temp_session = "_".join(temp_session[1:4])
        else:
            print('Recording type not specified. Skipping;...')
            return
        breakpoint_offset_idx = cur_breakpoint_df.index[
            cur_breakpoint_df['Session_file'].str.contains(temp_session)]
        try:
            breakpoint_offset = cur_breakpoint_df.loc[
                breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint

        except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file; no breakpoint offset needed
            if recording_type == 'synapse':
                breakpoint_offset = breakpoint_offset / 24414.0625
            elif recording_type == 'intan':
                breakpoint_offset = breakpoint_offset / 30000
            else:
                print('Recording type not specified. Skipping;...')
                return

        # Load key file
        key_times = read_csv(key_path)
        # Hard code here to eliminate all trials above 75 for SUBJ 174
        if subject_id == 'SUBJ-ID-174':
            key_times = key_times[key_times['TrialID'] < 75]

        for stim in sorted(list(set(key_times['AMdepth']))):
            # skip no-gos
            # if stim == 0:
            #     continue

            # Set up figure and axs
            plt.clf()
            f = plt.figure()
            ax_psth = f.add_subplot(212)
            ax_raster = f.add_subplot(211, sharex=ax_psth)
            if gaussian_window is not None:
                ax_gaussian = ax_psth.twinx()
            else:
                ax_gaussian = None

            cur_key_times = key_times[round(key_times['AMdepth'], 2) == round(stim, 2)
                                      ]['Trial_onset']

            # Populate axs
            common_psth_engine(spike_times=spike_times,
                               key_times=cur_key_times,
                               pre_stimulus_raster=pre_stimulus_raster,
                               post_stimulus_raster=post_stimulus_raster,
                               pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                               gaussian_window=gaussian_window,
                               ax_psth=ax_psth, ax_raster=ax_raster,
                               hist_bin_size_ms=10,
                               breakpoint_offset=breakpoint_offset,
                               do_plot=True,
                               rasterize=False)

            # Format axs
            format_ax(ax_raster)
            format_ax(ax_psth)

            ax_raster.axis('off')

            ax_psth.set_ylim([0, psth_max_ylim])
            ax_raster.set_ylim([-0.5, raster_max_y_lim])
            if gaussian_window is not None:
                ax_gaussian.set_ylim([-1, 1])
            ax_psth.set_ylabel("Spike rate by trial (Hz)")
            ax_psth.set_xlabel("Time (s)")
            f.suptitle("_".join(split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[0:4]) +
                       "_" + split("_*_", split(REGEX_SEP, memory_paths[0])[-1][:-4])[-1] + ":\n" +
                       str(round(stim, 2)) + " AM depth")

            if OUTPUT_TYPE == 'pdf':
                pdf.savefig()
            else:
                if recording_type == 'intan':
                    f.savefig(output_name + "_" + split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[3] + "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)
                else:
                    f.savefig(output_name + "_" +
                              "".join(split("-*-", split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[1])[1:3]) + "_" + str(
                        round(stim, 2)) + '.eps', format='eps', dpi=600)
            plt.close()

    if OUTPUT_TYPE == 'pdf':
        pdf.close()


def plot_psth_AMdepth_HitVsMiss(memory_paths,
                                all_key_path,
                                pre_stimulus_raster, post_stimulus_raster,
                                recording_type,  # synapse or intan
                                hist_bin_size_ms=10,
                                pre_stimulus_baseline=0.5,
                                cur_breakpoint_df=None,
                                uniform_psth_y_lim=True,
                                raster_max_y_lim=30.5):
    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = sep.join([OUTPUT_PATH, subject_id, cur_date, "Hit vs Miss"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys
    key_paths = glob(all_key_path + sep + subject_id + '*' +
                     cur_date + "*_trialInfo.csv")

    if len(key_paths) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths = glob(all_key_path + sep + subject_id + '*' +
                         cur_date + "*_trialInfo.csv")

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + sep + output_name + '_AMdepth.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    # Get max y lim for PSTHs across conditions (e.g. Pre, Active, Post) for the current unit
    psth_max_ylim = None
    if uniform_psth_y_lim:
        psth_max_ylim = 0
        for key_idx, key_path in enumerate(key_paths):
            temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
            if recording_type == 'synapse':
                temp_session = temp_session[1]
            elif recording_type == 'intan':
                temp_session = "_".join(temp_session[1:4])
            else:
                print('Recording type not specified. Skipping;...')
                return
            breakpoint_offset_idx = cur_breakpoint_df.index[
                cur_breakpoint_df['Session_file'].str.contains(temp_session)]
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
            except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
                try:
                    breakpoint_offset = cur_breakpoint_df.loc[
                        breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
                except KeyError:
                    breakpoint_offset = 0  # first file; no breakpoint offset needed
                if recording_type == 'synapse':
                    breakpoint_offset = breakpoint_offset / 24414.0625
                elif recording_type == 'intan':
                    breakpoint_offset = breakpoint_offset / 30000
                else:
                    print('Recording type not specified. Skipping;...')
                    return

            # Load key file
            key_times = read_csv(key_path)
            for stim in sorted(list(set(key_times['AMdepth']))):
                # skip no-gos
                if stim == 0:
                    continue
                cur_key_times = key_times[round(key_times['AMdepth'], 2) == round(stim, 2)]['Trial_onset']

                ## do something really inefficient here to hold y_axis uniform across all stimuli and treatments
                hist, _ = common_psth_engine(spike_times,
                                             cur_key_times,
                                             pre_stimulus_raster, post_stimulus_raster,
                                             breakpoint_offset=breakpoint_offset,
                                             hist_bin_size_ms=hist_bin_size_ms,
                                             do_plot=False)
                if np.max(hist) > psth_max_ylim:
                    psth_max_ylim = np.max(hist)

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            temp_session = split("_*_", split(REGEX_SEP, key_path)[-1])
            if recording_type == 'synapse':
                temp_session = temp_session[1]
            elif recording_type == 'intan':
                temp_session = "_".join(temp_session[1:4])
            else:
                print('Recording type not specified. Skipping;...')
                return
            breakpoint_offset_idx = cur_breakpoint_df.index[
                cur_breakpoint_df['Session_file'].str.contains(temp_session)]
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
            except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
                try:
                    breakpoint_offset = cur_breakpoint_df.loc[
                        breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
                except KeyError:
                    breakpoint_offset = 0  # first file; no breakpoint offset needed
                if recording_type == 'synapse':
                    breakpoint_offset = breakpoint_offset / 24414.0625
                elif recording_type == 'intan':
                    breakpoint_offset = breakpoint_offset / 30000
                else:
                    print('Recording type not specified. Skipping;...')
                    return

            # Load key file
            key_times = read_csv(key_path)

            for stim in sorted(list(set(key_times['AMdepth']))):
                # skip no-gos and Passive
                if "Aversive" not in key_path and "Active" not in key_path:  # skip passive stuff
                    continue
                if stim == 0:
                    continue

                # Set up figure and axs
                plt.clf()
                f = plt.figure()
                ax_psth_hit = f.add_subplot(223)
                ax_raster_hit = f.add_subplot(221, sharex=ax_psth_hit)
                ax_psth_miss = f.add_subplot(224)
                ax_raster_miss = f.add_subplot(222, sharex=ax_psth_miss)

                # Let's start with hits
                cur_stim_timestamps = key_times[round(key_times['AMdepth'], 2) == round(stim, 2)]
                timestamps = cur_stim_timestamps[cur_stim_timestamps['Hit'] == 1]['Trial_onset']

                # Populate axs
                common_psth_engine(spike_times=spike_times,
                                   key_times=timestamps,
                                   pre_stimulus_raster=pre_stimulus_raster,
                                   post_stimulus_raster=post_stimulus_raster,
                                   pre_stimulus_baseline=pre_stimulus_baseline,
                                   ax_psth=ax_psth_hit, ax_raster=ax_raster_hit,
                                   hist_bin_size_ms=10,
                                   breakpoint_offset=breakpoint_offset,
                                   do_plot=True)

                # Now misses
                timestamps = cur_stim_timestamps[cur_stim_timestamps['Miss'] == 1]['Trial_onset']

                # Populate axs
                common_psth_engine(spike_times=spike_times,
                                   key_times=timestamps,
                                   pre_stimulus_raster=pre_stimulus_raster,
                                   post_stimulus_raster=post_stimulus_raster,
                                   pre_stimulus_baseline=pre_stimulus_baseline,
                                   ax_psth=ax_psth_miss, ax_raster=ax_raster_miss,
                                   hist_bin_size_ms=10,
                                   breakpoint_offset=breakpoint_offset,
                                   do_plot=True)

                # Format axs
                format_ax(ax_raster_hit)
                format_ax(ax_psth_hit)
                format_ax(ax_raster_miss)
                format_ax(ax_psth_miss)

                ax_raster_hit.axis('off')
                ax_raster_miss.axis('off')

                ax_psth_hit.set_ylim([0, psth_max_ylim])
                ax_raster_hit.set_ylim([-0.5, raster_max_y_lim])
                ax_psth_hit.set_ylabel("Spike rate by trial (Hz)")
                ax_psth_hit.set_xlabel("Time (s)")

                ax_psth_miss.set_ylim([0, psth_max_ylim])
                ax_raster_miss.set_ylim([-0.5, raster_max_y_lim])
                ax_psth_miss.set_ylabel("")
                ax_psth_miss.set_xlabel("Time (s)")

                ax_raster_hit.set_title("Hit")
                ax_raster_miss.set_title("Miss")

                f.suptitle("_".join(split("_*_", split(REGEX_SEP, key_path)[-1][:-4])[0:2]) +
                           "_" + split("_*_", split(REGEX_SEP, memory_paths[0])[-1][:-4])[-1] + ":\n" +
                           str(round(stim, 2)) + " AM depth")

                pdf.savefig()

                # f.savefig(output_path + sep + "_".join(split("_*_", split(REGEX_SEP, key_name)[-1][:-4])[0:2]) +
                #           "_" + split("_*_", split(REGEX_SEP, memory_name)[-1][:-4])[-1] +
                #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

                plt.close()



def run_multiprocessing_plot(input_list, output_type='pdf'):
    memory_paths, pre_stimulus_raster, post_stimulus_raster, \
    wav_files_path, output_path, breakpoint_file_path, recording_type_dict = input_list

    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    subj_id = split("_*_", unit_id)[0]
    recording_type = recording_type_dict[subj_id]

    print("\nGenerating PSTHs for: " + unit_id)

    try:
        cur_breakpoint_file = glob(breakpoint_file_path + sep +
                                   "_".join(split_timestamps_name[0:3]) + "_breakpoints.csv")[0]
        cur_breakpoint_df = read_csv(cur_breakpoint_file)
    except IndexError:
        print("Breakpoint file not found for " + unit_id + ". Assuming non-concatenated...")
        cur_breakpoint_df = DataFrame()

    # For debugging purposes
    # if split_memory_name[4] in ['cluster173', 'cluster195']:
    #     return

    gaussian_window = None
    #
    # '''
    # Plot spout-offset PSTH
    # '''
    # # if capping trials at 500
    # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 50.
    # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 50.
    # plot_psth_spoutOffset(memory_paths,
    #                       KEYS_PATH,
    #                       pre_stimulus_raster, post_stimulus_raster,
    #                       recording_type,
    #                       gaussian_window=gaussian_window,
    #                       hist_bin_size_ms=10,
    #                       cur_breakpoint_df=cur_breakpoint_df,
    #                       uniform_psth_y_lim=False,
    #                       raster_max_y_lim=500.5)
    #
    # '''
    # Plot spout-onset PSTH
    # '''
    # # if capping trials at 500
    # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 50.
    # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 50.
    # plot_psth_spoutOnset(memory_paths,
    #                      KEYS_PATH,
    #                      pre_stimulus_raster, post_stimulus_raster,
    #                      recording_type,
    #                      gaussian_window=gaussian_window,
    #                      hist_bin_size_ms=10,
    #                      cur_breakpoint_df=cur_breakpoint_df,
    #                      uniform_psth_y_lim=False,
    #                      raster_max_y_lim=500.5)

    '''
    Plot AM-depth PSTH
    '''
    # if capping raster trials at 100
    mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 8.
    mpl.rcParams['lines.markeredgewidth'] = mpl.rcParams['lines.markersize'] / 2
    plot_psth_AMdepth(memory_paths,
                      KEYS_PATH,
                      pre_stimulus_raster, post_stimulus_raster,
                      recording_type,
                      gaussian_window=gaussian_window,
                      hist_bin_size_ms=10,
                      cur_breakpoint_df=cur_breakpoint_df,
                      uniform_psth_y_lim=True,
                      raster_max_y_lim=100.5,
                      override_max_ylim=50.5)

    # '''
    # Plot AM-depth PSTH by trial response (Hit vs Miss)
    # '''
    # # if capping trials at 30
    # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 2.
    # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 10.
    # plot_psth_AMdepth_HitVsMiss(memory_paths,
    #                             KEYS_PATH,
    #                             pre_stimulus_raster, post_stimulus_raster,
    #                             recording_type,
    #                             hist_bin_size_ms=10,
    #                             cur_breakpoint_df=cur_breakpoint_df,
    #                             uniform_psth_y_lim=True,
    #                             raster_max_y_lim=30.5)


"""
Set global paths and variables
"""
warnings.filterwarnings("ignore")

SPIKES_PATH = '.' + sep + sep.join(['Data', 'Spike times'])
KEYS_PATH = '.' + sep + sep.join(['Data', 'Key files'])
OUTPUT_PATH = '.' + sep + sep.join(['Data', 'Output', 'PSTHs new'])
BREAKPOINT_PATH = '.' + sep + sep.join(['Data', 'Breakpoints'])
OUTPUT_TYPE = 'pdf'

WAV_FILES_PATH = r'./Data/Stimuli'  # TODO: design AM stimuli .wav

# SAMPLING_RATE = 30000

# Only run these cells/su or None to run all

# ActiveBaseline Decrease, increase, and none representatives
# CELLS_TO_RUN = ['SUBJ-ID-154_210511_concat_cluster1267',
#                 'SUBJ-ID-151_210510_concat_cluster1466',
#                 'SUBJ-ID-231_210710_concat_cluster572']
CELLS_TO_RUN = None

SUBJECTS_TO_RUN = None

RECORDING_TYPE_DICT = {
    'SUBJ-ID-197': 'synapse',
    'SUBJ-ID-151': 'synapse',
    'SUBJ-ID-154': 'synapse',
    'SUBJ-ID-231': 'intan',
    'SUBJ-ID-232': 'intan',
}

PRE_STIMULUS_RASTER = 0
POST_STIMULUS_RASTER = 1.
NUMBER_OF_CORES = 4

# Set plotting parameters
LABEL_FONT_SIZE = 15
TICK_LABEL_SIZE = 10
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.size'] = LABEL_FONT_SIZE * 1.5
mpl.rcParams['axes.labelsize'] = LABEL_FONT_SIZE * 1.5
mpl.rcParams['axes.titlesize'] = LABEL_FONT_SIZE
mpl.rcParams['axes.linewidth'] = LABEL_FONT_SIZE / 12.
mpl.rcParams['legend.fontsize'] = LABEL_FONT_SIZE / 2.
mpl.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE * 1.5
mpl.rcParams['ytick.labelsize'] = TICK_LABEL_SIZE * 1.5
mpl.rcParams['errorbar.capsize'] = LABEL_FONT_SIZE
mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 30.
mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 30.
mpl.rcParams['lines.linewidth'] = LABEL_FONT_SIZE / 8.

if __name__ == '__main__':
    # Generate a list of inputs to be passed to each worker
    input_lists = list()

    makedirs(OUTPUT_PATH, exist_ok=True)

    memory_paths = glob(SPIKES_PATH + sep + '*_cluster*.txt')

    # group them in list of lists based on unit name
    keyf = lambda text: text.split("_")[-1][:-4]
    # grouped_memory_paths = [list(list([s for s in memory_paths if keyf(item) in s])) for item in memory_paths]
    # Feed each worker with all memory paths from one unit
    for unit_path in memory_paths:
        if CELLS_TO_RUN is not None:
            if any([chosen for chosen in CELLS_TO_RUN if chosen in unit_path]):
                pass
            else:
                continue

        if SUBJECTS_TO_RUN is not None:
            if any([chosen for chosen in SUBJECTS_TO_RUN if chosen in unit_path]):
                pass
            else:
                continue

        input_lists.append(([unit_path], PRE_STIMULUS_RASTER, POST_STIMULUS_RASTER, WAV_FILES_PATH,
                            OUTPUT_PATH, BREAKPOINT_PATH, RECORDING_TYPE_DICT))

    pool = mp.Pool(NUMBER_OF_CORES)

    # Feed each worker with all memory paths from one unit
    pool.map(run_multiprocessing_plot, input_lists)

    pool.close()
    pool.join()
