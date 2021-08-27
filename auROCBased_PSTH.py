from intrastim_correlation import *
import matplotlib as mpl
from pandas import read_csv, DataFrame

from format_axes import *
from matplotlib.backends.backend_pdf import PdfPages

import warnings
import multiprocessing as mp
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel
from os.path import sep
import platform

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def gaussian_moving_average_fft(points, sigma):
    return convolve_fft(points, Gaussian1DKernel(stddev=sigma))


def auROC_response_curve(hist, edges, pre_stimulus_baseline, window=0.1):
    """
    :param hist:
    :param window_ms:
    :return:
    """
    # Baseline from -2 to -1 s
    baseline_points_mask = (edges >= -pre_stimulus_baseline) & (edges < -1)
    baseline_hist = hist[baseline_points_mask[:-1]]

    # For every bin during response
    auroc_curve = []
    for start_bin in np.arange(edges[0], edges[-1], window):
        cur_points_mask = (edges >= start_bin) & (edges < start_bin+window)
        cur_hist_values = hist[cur_points_mask[:-1]]

        max_criterion = np.max(np.concatenate((baseline_hist, cur_hist_values), axis=None))
        if max_criterion > 0:
            thresholds = np.linspace(0, max_criterion, int(max_criterion/0.1), endpoint=True)
        else:
            thresholds = [0, 1]  # Fix for when there's zero spikes to still get auROC=0.5

        false_positive = []
        true_positive = []
        for t in thresholds:
            response_above_t = cur_hist_values >= t
            baseline_above_t = baseline_hist >= t

            false_positive.append(sum(baseline_above_t)/len(baseline_hist))
            true_positive.append(sum(response_above_t)/len(cur_hist_values))
        auroc_curve.append(np.trapz(sorted(true_positive), sorted(false_positive)))
        # # For debugging
        # mpl.use('TkAgg')
        # plt.figure()
        # plt.plot(false_positive, true_positive)
        # plt.show()

    return auroc_curve


def common_psth_engine(spike_times,
                       key_times,
                       sampling_rate,
                       pre_stimulus_raster, post_stimulus_raster,
                       pre_stimulus_baseline=None, sliding_window=None,
                       ax_raster=None, ax_psth=None, ax_gaussian=None,
                       breakpoint_offset=None,
                       hist_bin_size_ms=10,
                       do_plot=True):
    number_of_stimulus_repetitions = len(key_times)

    bin_size = hist_bin_size_ms / 1000  # in s
    bin_cuts = np.arange(-pre_stimulus_raster, post_stimulus_raster, bin_size)

    # Loop through each stimulus presentation
    raster_trial_counter = number_of_stimulus_repetitions
    relative_times = list()
    for cur_stim_time in key_times:
        offset_stim_time = cur_stim_time + breakpoint_offset / sampling_rate

        # Get spike times around the current stimulus onset
        times_to_plot = spike_times[((offset_stim_time - pre_stimulus_raster) < spike_times) &
                                    (spike_times < (offset_stim_time + post_stimulus_raster))]

        # Zero-center spike times
        curr_relative_times = times_to_plot - offset_stim_time
        if do_plot:
            ax_raster.plot(curr_relative_times,
                           np.repeat(raster_trial_counter, len(curr_relative_times)),
                           'k|',
                           rasterized=True)
            raster_trial_counter -= 1

        relative_times.append([x for x in curr_relative_times])

    relative_times = [item for sublist in relative_times for item in sublist]

    # Only to get y labels. Later, use this to plot as well
    hist, edges = np.histogram(relative_times, bins=bin_cuts)

    # Change hist to spike rate before appending
    hist = np.round(hist / number_of_stimulus_repetitions / bin_size, 2)


    # From Cohen et al., Nature, 2012
    # a, Raster plot from 15 trials of 149 big-reward trials from a dopaminergic
    # neuron. r1 and r2 correspond to two example 100-ms bins. b, Average firing rate of this neuron. c, Area
    # under the receiver operating characteristic curve (auROC) for r1, in which the neuron increased its firing
    # rate relative to baseline. We compared the histogram of spike counts during the baseline period (dashed
    # line) to that during a given bin (solid line) by moving a criterion from zero to the maximum firing rate (in
    # this example, 68 spikes/s). We then plotted the probability that the activity during r1 was greater than the
    # criteria against the probability that the baseline activity was greater than the criteria. The area under this
    # curve quantifies the degree of overlap between the two spike count distributions (i.e., the discriminability
    # of the two).
    auroc_curve = None
    if sliding_window is not None:
        auroc_curve = auROC_response_curve(hist, edges, pre_stimulus_baseline, sliding_window)

    if do_plot:
        ax_psth.bar(bin_cuts[:-1], hist, 0.01, color='k')
        # ax_psth.hist(relative_times, bins=bin_cuts, edgecolor=None, facecolor='k')

        if sliding_window is not None:
            # gaussian_psth_y = gaussian_moving_average_fft(auroc_curve, 3)
            gaussian_psth_y = auroc_curve
            gaussian_psth_x = np.linspace(-pre_stimulus_raster, post_stimulus_raster, len(gaussian_psth_y))
            ax_gaussian.plot(gaussian_psth_x, gaussian_psth_y, color='steelblue')

    return hist, auroc_curve


def plot_psth_spoutOffset(memory_paths,
                          all_key_path,
                          sampling_rate,
                          pre_stimulus_raster, post_stimulus_raster,
                          hist_bin_size_ms=10,
                          pre_stimulus_baseline=0.5,
                          sliding_window=5.,
                          cur_breakpoint_df=DataFrame(),
                          uniform_psth_y_lim=False,
                          raster_max_y_lim=500.5):
    split_memory_path = split("\\\\", memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = "\\".join([OUTPUT_PATH, subject_id, cur_date, "Hit spout offset"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys
    try:
        spout_key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_spoutTimestamps.csv")
        info_key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_trialInfo.csv")
    except IndexError:
        print("Key not found for " + unit_id + ", session date: " + cur_date)
        return

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in spout_key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    spout_key_paths = [key_path for _, key_path in sorted(zip(session_times, spout_key_paths), key=lambda pair: pair[0])]

    split_key_paths = [split("_*_", key_path) for key_path in info_key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    info_key_paths = [key_path for _, key_path in sorted(zip(session_times, info_key_paths), key=lambda pair: pair[0])]


    output_name_pdf = output_subfolder + "\\" + output_name + '_spoutOffsetHits.pdf'
    output_name_csv = output_subfolder + "\\" + output_name + '_spoutOffsetHits_auROC.csv'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, dummy_key_path in enumerate(spout_key_paths):
            if "Aversive" not in dummy_key_path and "Active" not in dummy_key_path:  # skip passive stuff
                continue
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

            # Load key file
            spout_key_times = read_csv(spout_key_paths[key_idx])
            info_key_times = read_csv(info_key_paths[key_idx])

            # Select spout-offset times that trigger a Hit
            hit_timestamps = info_key_times[info_key_times['Hit'] == 1]
            # Grab the last spout offset in a trial (the one triggering the hit)
            selected_key_times = []
            for _, cur_timestamp in hit_timestamps.iterrows():
                curr_offset = spout_key_times[
                    (spout_key_times['Spout_offset'] < cur_timestamp['Trial_offset']) &
                    (spout_key_times['Spout_offset'] > cur_timestamp['Trial_onset'])]['Spout_offset'].values
                # If there's more than one offset value, grab always the last
                selected_key_times.append(curr_offset[-1])

            # Set up figure and axs
            plt.clf()
            f = plt.figure()
            ax_psth = f.add_subplot(212)
            ax_raster = f.add_subplot(211, sharex=ax_psth)
            ax_gaussian = ax_psth.twinx()
            # Populate axs
            _, auroc_curve = common_psth_engine(spike_times=spike_times,
                               key_times=selected_key_times,
                               sampling_rate=sampling_rate,
                               pre_stimulus_raster=pre_stimulus_raster,
                               post_stimulus_raster=post_stimulus_raster,
                               ax_psth=ax_psth, ax_raster=ax_raster,
                               pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                               sliding_window=sliding_window,
                               hist_bin_size_ms=10,
                               breakpoint_offset=breakpoint_offset,
                               do_plot=True)

            # Format axs
            format_ax(ax_raster)
            format_ax(ax_psth)

            ax_raster.axis('off')

            ax_raster.set_ylim([-0.5, raster_max_y_lim])
            ax_gaussian.set_ylim([0, 1])

            ax_psth.set_ylabel("Spike rate by trial (Hz)")
            ax_psth.set_xlabel("Time (s)")
            ax_gaussian.set_ylabel("Spike auROC vs baseline")
            f.suptitle("_".join(split("_*_", split("\\\\", dummy_key_path)[-1][:-4])[0:2]) +
                       "_" + split("_*_", split("\\\\", memory_paths[0])[-1][:-4])[-1] + ":\n" + "Spout offset")

            pdf.savefig()

            # f.savefig(output_path + '\\' + "_".join(split("_*_", split("\\\\", key_name)[-1][:-4])[0:2]) +
            #           "_" + split("_*_", split("\\\\", memory_name)[-1][:-4])[-1] +
            #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

            plt.close()

    # Write auROC points
    with open(output_name_csv, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Unit'] + ['Time'] + ['auROC'])

        gaussian_psth_x = np.linspace(-pre_stimulus_raster, post_stimulus_raster, len(auroc_curve))

        for dummy_idx in range(0, len(auroc_curve)):
            writer.writerow([unit_id] + [gaussian_psth_x[dummy_idx]] + [auroc_curve[dummy_idx]])


def plot_psth_spoutOnset(memory_paths,
                         all_key_path,
                         sampling_rate,
                         pre_stimulus_raster, post_stimulus_raster,
                         hist_bin_size_ms=10,
                         pre_stimulus_baseline=0.5,
                         sliding_window=5,
                         cur_breakpoint_df=None,
                         uniform_psth_y_lim=False,
                         raster_max_y_lim=500.5):
    split_memory_path = split("\\\\", memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = "\\".join([OUTPUT_PATH, subject_id, cur_date, "Spout onset"])
    makedirs(output_subfolder, exist_ok=True)
    # Use subj-session date to grab appropriate keys
    try:
        key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_spoutTimestamps.csv")
    except IndexError:
        print("Key not found for " + unit_id + ", session date: " + cur_date)
        return

    if len(key_paths) == 0:
        key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_spoutTimestamps.csv")

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + "\\" + output_name + '_spoutOnset.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            if "Aversive" not in key_path and "Active" not in key_path:  # skip passive stuff
                continue
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

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
                               sampling_rate=sampling_rate,
                               pre_stimulus_raster=pre_stimulus_raster,
                               post_stimulus_raster=post_stimulus_raster,
                               ax_psth=ax_psth, ax_raster=ax_raster,
                               pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                               sliding_window=sliding_window,
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
            f.suptitle("_".join(split("_*_", split("\\\\", key_path)[-1][:-4])[0:2]) +
                       "_" + split("_*_", split("\\\\", memory_paths[0])[-1][:-4])[-1] + ":\n" + "Spout onset")

            pdf.savefig()

            # f.savefig(output_path + '\\' + "_".join(split("_*_", split("\\\\", key_name)[-1][:-4])[0:2]) +
            #           "_" + split("_*_", split("\\\\", memory_name)[-1][:-4])[-1] +
            #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

            plt.close()


def plot_psth_AMdepth(memory_paths,
                      all_key_path,
                      sampling_rate,
                      pre_stimulus_raster, post_stimulus_raster,
                      hist_bin_size_ms=10,
                      pre_stimulus_baseline=0.5,
                      gaussian_window=5,
                      cur_breakpoint_df=None,
                      uniform_psth_y_lim=True,
                      raster_max_y_lim=30.5):
    split_memory_path = split("\\\\", memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = "\\".join([OUTPUT_PATH, subject_id, cur_date, "AM depth"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys
    try:
        key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_trialInfo.csv")
    except IndexError:
        print("Key not found for " + unit_id + ", session date: " + cur_date)
        return

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + "\\" + output_name + '_AMdepth.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    # Get max y lim for PSTHs across conditions (e.g. Pre, Active, Post) for the current unit
    psth_max_ylim = None
    if uniform_psth_y_lim:
        psth_max_ylim = 0
        for key_idx, key_path in enumerate(key_paths):
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

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
                                             sampling_rate,
                                             pre_stimulus_raster, post_stimulus_raster,
                                             breakpoint_offset=breakpoint_offset,
                                             hist_bin_size_ms=hist_bin_size_ms,
                                             do_plot=False)
                if np.max(hist) > psth_max_ylim:
                    psth_max_ylim = np.max(hist)

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

            # Load key file
            key_times = read_csv(key_path)

            for stim in sorted(list(set(key_times['AMdepth']))):
                # skip no-gos
                if stim == 0:
                    continue

                # Set up figure and axs
                plt.clf()
                f = plt.figure()
                ax_psth = f.add_subplot(212)
                ax_raster = f.add_subplot(211, sharex=ax_psth)
                ax_gaussian = ax_psth.twinx()

                cur_key_times = key_times[round(key_times['AMdepth'], 2) == round(stim, 2)
                                          ]['Trial_onset']

                # Populate axs
                common_psth_engine(spike_times=spike_times,
                                   key_times=cur_key_times,
                                   sampling_rate=sampling_rate,
                                   pre_stimulus_raster=pre_stimulus_raster,
                                   post_stimulus_raster=post_stimulus_raster,
                                   pre_stimulus_baseline=pre_stimulus_baseline, ax_gaussian=ax_gaussian,
                                   sliding_window=gaussian_window,
                                   ax_psth=ax_psth, ax_raster=ax_raster,
                                   hist_bin_size_ms=10,
                                   breakpoint_offset=breakpoint_offset,
                                   do_plot=True)

                # Format axs
                format_ax(ax_raster)
                format_ax(ax_psth)

                ax_raster.axis('off')

                ax_psth.set_ylim([0, psth_max_ylim])
                ax_raster.set_ylim([-0.5, raster_max_y_lim])
                ax_gaussian.set_ylim([-1, 1])
                ax_psth.set_ylabel("Spike rate by trial (Hz)")
                ax_psth.set_xlabel("Time (s)")
                f.suptitle("_".join(split("_*_", split("\\\\", key_path)[-1][:-4])[0:2]) +
                           "_" + split("_*_", split("\\\\", memory_paths[0])[-1][:-4])[-1] + ":\n" +
                           str(round(stim, 2)) + " AM depth")

                pdf.savefig()

                # f.savefig(output_path + '\\' + "_".join(split("_*_", split("\\\\", key_name)[-1][:-4])[0:2]) +
                #           "_" + split("_*_", split("\\\\", memory_name)[-1][:-4])[-1] +
                #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

                plt.close()


def plot_psth_AMdepth_HitVsMiss(memory_paths,
                                all_key_path,
                                sampling_rate,
                                pre_stimulus_raster, post_stimulus_raster,
                                hist_bin_size_ms=10,
                                pre_stimulus_baseline=0.5,
                                cur_breakpoint_df=None,
                                uniform_psth_y_lim=True,
                                raster_max_y_lim=30.5):
    split_memory_path = split("\\\\", memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]

    output_name = unit_id + "_PSTH_10ms"

    output_subfolder = "\\".join([OUTPUT_PATH, subject_id, cur_date, "Hit vs Miss"])
    makedirs(output_subfolder, exist_ok=True)

    # Use subj-session date to grab appropriate keys
    try:
        key_paths = glob(all_key_path + '\\' + subject_id + '*' +
                         cur_date + "*_trialInfo.csv")
    except IndexError:
        print("Key not found for " + unit_id + ", session date: " + cur_date)
        return

    # make sure they're ordered by session time
    split_key_paths = [split("_*_", key_path) for key_path in key_paths]
    session_times = [split("-*-", split_path[1])[-1] for split_path in split_key_paths]
    key_paths = [key_path for _, key_path in sorted(zip(session_times, key_paths), key=lambda pair: pair[0])]

    output_name_pdf = output_subfolder + "\\" + output_name + '_AMdepth.pdf'

    # Load spike times
    spike_times = np.genfromtxt(memory_paths[0])

    # Get max y lim for PSTHs across conditions (e.g. Pre, Active, Post) for the current unit
    psth_max_ylim = None
    if uniform_psth_y_lim:
        psth_max_ylim = 0
        for key_idx, key_path in enumerate(key_paths):
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

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
                                             sampling_rate,
                                             pre_stimulus_raster, post_stimulus_raster,
                                             breakpoint_offset=breakpoint_offset,
                                             hist_bin_size_ms=hist_bin_size_ms,
                                             do_plot=False)
                if np.max(hist) > psth_max_ylim:
                    psth_max_ylim = np.max(hist)

    with PdfPages(output_name_pdf) as pdf:
        for key_idx, key_path in enumerate(key_paths):
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    key_idx - 1, 'Break_point']  # grab previous session's breakpoint
            except KeyError:
                breakpoint_offset = 0  # first file

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
                                   sampling_rate=sampling_rate,
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
                                   sampling_rate=sampling_rate,
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

                f.suptitle("_".join(split("_*_", split("\\\\", key_path)[-1][:-4])[0:2]) +
                           "_" + split("_*_", split("\\\\", memory_paths[0])[-1][:-4])[-1] + ":\n" +
                           str(round(stim, 2)) + " AM depth")

                pdf.savefig()

                # f.savefig(output_path + '\\' + "_".join(split("_*_", split("\\\\", key_name)[-1][:-4])[0:2]) +
                #           "_" + split("_*_", split("\\\\", memory_name)[-1][:-4])[-1] +
                #           "_" + str(round(stim, 2)) + '.eps', format='eps', dpi=600)

                plt.close()


#
def run_multiprocessing_plot(input_list):
    memory_paths, pre_stimulus_raster, post_stimulus_raster, \
    output_path, breakpoint_file_path = input_list

    # Split path name to get subject, session and unit ID for prettier output

    split_memory_path = split(REGEX_SEP, memory_paths[0])  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    subj_id = split_memory_path[0]

    print("Generating PSTHs for: " + unit_id)

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

    sliding_window = 0.1
    #
    '''
    Plot spout-offset PSTH triggering a Hit
    '''
    # if capping trials at 500
    mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 50.
    mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 50.
    plot_psth_spoutOffset(memory_paths,
                          KEYS_PATH,
                          pre_stimulus_raster, post_stimulus_raster,
                          pre_stimulus_baseline=2,
                          sliding_window=sliding_window,
                          hist_bin_size_ms=10,
                          cur_breakpoint_df=cur_breakpoint_df,
                          uniform_psth_y_lim=False,
                          raster_max_y_lim=500.5)
    #
    # '''
    # Plot spout-onset PSTH following a Hit
    # '''
    # # if capping trials at 500
    # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 50.
    # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 50.
    # plot_psth_spoutOnset(memory_paths,
    #                      ALL_KEYS_PATH,
    #                      SAMPLING_RATE,
    #                      pre_stimulus_raster, post_stimulus_raster,
    #                      sliding_window=sliding_window,
    #                      hist_bin_size_ms=10,
    #                      cur_breakpoint_df=cur_breakpoint_df,
    #                      uniform_psth_y_lim=False,
    #                      raster_max_y_lim=500.5)
    # #
    # # '''
    # # Plot AM-depth PSTH
    # # '''
    # # # if capping trials at 30
    # # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 2.
    # # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 10.
    # # plot_psth_AMdepth(memory_paths,
    # #                   ALL_KEYS_PATH,
    # #                   SAMPLING_RATE,
    # #                   pre_stimulus_raster, post_stimulus_raster,
    # #                   gaussian_window=sliding_window,
    # #                   hist_bin_size_ms=10,
    # #                   cur_breakpoint_df=cur_breakpoint_df,
    # #                   uniform_psth_y_lim=True,
    # #                   raster_max_y_lim=30.5)
    #
    # '''
    # Plot AM-depth PSTH by trial response (Hit vs Miss)
    # '''
    # # if capping trials at 30
    # mpl.rcParams['lines.markersize'] = LABEL_FONT_SIZE / 2.
    # mpl.rcParams['lines.markeredgewidth'] = LABEL_FONT_SIZE / 10.
    # plot_psth_AMdepth_HitVsMiss(memory_paths,
    #                             ALL_KEYS_PATH,
    #                             SAMPLING_RATE,
    #                             pre_stimulus_raster, post_stimulus_raster,
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
OUTPUT_PATH = '.' + sep + sep.join(['Data', 'Output'])
BREAKPOINT_PATH = '.' + sep + sep.join(['Data', 'Breakpoints'])

# WAV_FILES_PATH = r'.\Data\Stimuli'  # TODO: design AM stimuli .wav


# Only run these cells/su or None to run all
CELLS_TO_RUN = None
SUBJECTS_TO_RUN = None

PRE_STIMULUS_RASTER = 2
POST_STIMULUS_RASTER = 4
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

    memory_paths = glob(SPIKES_PATH + '\*_cluster*.txt')

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

        input_lists.append(([unit_path], PRE_STIMULUS_RASTER, POST_STIMULUS_RASTER,
                            OUTPUT_PATH, BREAKPOINT_PATH))

    pool = mp.Pool(NUMBER_OF_CORES)

    # Feed each worker with all memory paths from one unit
    pool.map(run_multiprocessing_plot, input_lists)

    pool.close()
    pool.join()
