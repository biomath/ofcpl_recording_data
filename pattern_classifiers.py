__author__ = 'Matheus'

import numpy as np
from random import shuffle, choice
from itertools import chain
from scipy.stats import gaussian_kde, norm, ks_2samp, ttest_ind
import logging
from fastdtw import fastdtw
import matplotlib as mpl
mpl.use('pdf')
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import resample
import process_spike_data as psd
import matplotlib.ticker as ticker
# import objgraph as og


def rcorr(g1, g2):
    # When correlating two flat responses, rcorr yields 'nan'; this "corrects" it
    if (np.sum(g1) == 0) and (np.sum(g2) == 0):
        return np.nan
    elif (np.sum(g1) == 0) or (np.sum(g2) == 0):
        return np.nan
    else:
        return np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))


def normalize_vector(g):
    if np.sum(g) != 0:
        return (g - np.min(g)) / (np.max(g) - np.min(g))
    else:
        return g


def cross_corr(g1, g2):
    if (np.sum(g1) == 0) and (np.sum(g2) == 0):
        return 1
    elif (np.sum(g1) == 0) or (np.sum(g2) == 0):
        return 0
    else:
        return np.corrcoef(g1, g2)[1, 0]


def dtw_nearness(g1, g2, radius):
    ret_val, _ = fastdtw(g1, g2, radius=radius)
    try:
        ret_val = 1.0 / ret_val
    except ZeroDivisionError:
        ret_val = 1
    return ret_val


def corr_function(g1, g2, correlation_method, dtw_radius=1, normalize=True):
    if correlation_method is 'count':
        return np.absolute(np.sum(g1) - np.sum(g2))

    elif normalize:
        g1 = normalize_vector(g1)
        g2 = normalize_vector(g2)
        if correlation_method is 'rcorr':
            return rcorr(g1, g2)
        elif correlation_method is 'cross_corr':
            return cross_corr(g1, g2)
        elif correlation_method is 'dtw':
            ret_val = dtw_nearness(g1, g2, radius=dtw_radius)
            return ret_val

    else:
        if correlation_method is 'rcorr':
            return rcorr(g1, g2)
        elif correlation_method is 'cross_corr':
            return cross_corr(g1, g2)
        elif correlation_method is 'dtw':
            ret_val, _ = fastdtw(g1, g2, radius=dtw_radius)
            return ret_val


def generate_psth(spike_times, key_times, psth_bin_size_ms, stim_duration):
    bin_size = psth_bin_size_ms / 1000  # in s
    bin_cuts = np.arange(0, stim_duration, bin_size)

    # Dictionary format: {stim_code: (relative_times, bin_cuts)}
    ret_dict = {}
    for stim in set(key_times.iloc[:, 1]):
        relative_times = list()
        for curr_stim_time in key_times[key_times.iloc[:, 1] == stim].iloc[:, 0]:
            times_to_plot = spike_times[(curr_stim_time < spike_times) &
                                        (spike_times < (curr_stim_time + stim_duration))]
            curr_relative_times = times_to_plot - curr_stim_time
            relative_times.append([x for x in curr_relative_times])
        relative_times = [item for sublist in relative_times for item in sublist]
        ret_dict.update({stim: (relative_times, bin_cuts)})

    return ret_dict


def extraction_index(song_psth, scene_psth, chorus_psth, method='rcorr', sampling_rate=0.001, normalize=True):
    # smooth PSTH first
    # Schneider & Woolley 2013 use a 5 ms sliding Hanning filter on their PSTH. Using Gaussian here
    # They are not specific about the binning for the PSTH used here. Let's try 1 ms with 5 ms smoothing
    song_psth_smoothed = psd.gaussian_moving_average_fft(song_psth, sampling_rate, 5, 'times')
    scene_psth_smoothed = psd.gaussian_moving_average_fft(scene_psth, sampling_rate, 5, 'times')
    chorus_psth_smoothed = psd.gaussian_moving_average_fft(chorus_psth, sampling_rate, 5, 'times')

    # Recommended so that firing rates don't skew too much the correlations
    if normalize:
        song_psth_smoothed = normalize_vector(song_psth_smoothed)
        scene_psth_smoothed = normalize_vector(scene_psth_smoothed)
        chorus_psth_smoothed = normalize_vector(chorus_psth_smoothed)

    r_song = corr_function(song_psth_smoothed, scene_psth_smoothed, method, normalize=normalize)
    r_chor = corr_function(scene_psth_smoothed, chorus_psth_smoothed, method, normalize=normalize)

    return (r_song - r_chor) / (r_song + r_chor)


def stim_classify(rasters_dict,
                  method,
                  brain_area,
                  count_bin=(2,),
                  sampling_rate=0.001,
                  downsample_q=None,
                  dtw_radius=None,
                  trial_shuffling=False,
                  simulations=1000):
    number_of_stimuli = len(rasters_dict.keys())
    # print("Stim classify...")
    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    rasters_dict = dict((key_train[0], key_train[1][0:number_of_stimulus_repetitions])
                        for key_train in rasters_dict.items())

    if trial_shuffling is True:
        counter_diagonal_accuracies = np.zeros(simulations)

    if method in ('rcorr', 'dtw', 'cross_corr'):
        # Loop runs until all combinations of trains have been used as templates
        # Use template 1 only as a proxy for all
        def test_template(train, template_set):
            def max_indices(lst):
                """
                This function returns the indices of all the maxima in the list
                """
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(max(lst), offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            temp = list()
            if method is 'rcorr':
                for template in template_set:
                    temp.append(rcorr(train, template))
            elif method is 'dtw':
                for template in template_set:
                    dtw_inv_distance = dtw_nearness(train, template, radius=dtw_radius)
                    temp.append(dtw_inv_distance)
            elif method is 'cross_corr':
                for template in template_set:
                    temp.append(cross_corr(train, template))

            index_max = max_indices(temp)

            # If there are ties, choose randomly
            if len(index_max) > 1:
                index_max = choice(index_max)

            return index_max

    else:
        def test_template(train, template_set):
            def min_indices(lst):
                """
                This function returns the indices of all the minima in the list
                """
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(min(lst), offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            temp = list()

            if method is 'count':
                for template in template_set:
                    # vector dot-subtraction (element-by-element, aka bin-by-bin)
                    #   then square all values (squared distances) then sum them
                    temp.append(np.sum(np.power(np.subtract(train, template), 2)))

            index_min = min_indices(temp)
            # If there are ties, choose randomly
            if len(index_min) > 1:
                index_min = choice(index_min)
            return index_min

    # Will hold the total average of classifications
    counter_accuracy_matrix = np.zeros([number_of_stimuli, number_of_stimuli])

    if downsample_q is not None:
        rasters_dict = dict((key_train[0], [resample(train, len(train)//downsample_q)
                                            for train in key_train[1]])
                            for key_train in rasters_dict.items())

    if method is 'count':
        rasters_dict = dict((key_train[0], [psd.bin_train(train, sampling_rate, count_bin)
                                            for train in key_train[1]])
                            for key_train in rasters_dict.items())

    for i in np.arange(0, simulations):
        # Creates copies of raster dictionary to pop templates from it
        temp_raster_dict = deepcopy(rasters_dict)

        template_list = list()
        for key in rasters_dict.keys():
            template_list.append(temp_raster_dict[key].pop(np.random.randint(0, number_of_stimulus_repetitions)))

        # Will hold accuracies in the current run of the classifier
        # rows are actual stimulus, columns are predicted
        current_matrix = np.zeros([number_of_stimuli, number_of_stimuli])

        # Take responses to each stimuli and compare to set of templates
        actual_stim_index = 0
        for key in rasters_dict.keys():
            for train in temp_raster_dict[key]:
                predicted_stim_index = test_template(train, template_list)
                current_matrix[actual_stim_index, predicted_stim_index] += 1
            actual_stim_index += 1

        # Sum performance of current templates and add to main matrix
        counter_accuracy_matrix += current_matrix

        if trial_shuffling is True:
            if brain_area is 'HVC':
                # Only one BOS
                counter_diagonal_accuracies[i] = current_matrix[0, 0]

            elif brain_area is 'NCM':
                # Compute means
                counter_diagonal_accuracies[i] = np.mean(np.diag(current_matrix))

    # Transform values from accuracy counts to averages
    # number_of_stimulus_repetitions - 1 explanation: one template is left out,
    #   so N - 1 comparisons are made per stimulus
    mean_accuracy_matrix = counter_accuracy_matrix / (simulations * number_of_stimulus_repetitions - 1)
    mean_diagonal_accuracies = counter_diagonal_accuracies / (number_of_stimulus_repetitions - 1)

    if trial_shuffling is True:
        return mean_accuracy_matrix, mean_diagonal_accuracies
    else:
        return mean_accuracy_matrix


def trial_shuffling_approach(rasters_dict, mapped_accuracies, method, brain_area, count_bin=(2,),
                             simulations=1000, downsample_q=None, dtw_radius=1):
    """
    This function is designed for HVC selectivity towards BOS. Therefore, accuracy is computed as an average of DS and
    UDS only
    Updated on 6/10/15: Optimized to run after the mapped classsification
    :param rasters_dict:
    :param accuracy_matrix:
    :param method:
    :param number_of_stimulus_repetitions:
    :param simulations:
    :return: significance, jumbled accuracies, mapped accuracies
    """

    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    # Making a flat list out of the dictionary
    jumbled_rasters = list(chain.from_iterable(rasters_dict.values()))

    shuffle(jumbled_rasters)

    number_of_stimuli = len(rasters_dict)
    # Randomization
    jumbled_rasters_dict = dict((key, []) for key in rasters_dict.keys())
    dict_keys = list(jumbled_rasters_dict.copy().keys())
    for i, train in enumerate(jumbled_rasters):
        jumbled_rasters_dict[dict_keys[i % number_of_stimuli]].append(train)  # distribute trains among keys of the dict

    # Classification
    _, jumbled_accuracies = stim_classify(jumbled_rasters_dict,
                                          brain_area=brain_area,
                                          method=method,
                                          count_bin=count_bin,
                                          downsample_q=downsample_q,
                                          dtw_radius=dtw_radius,
                                          trial_shuffling=True,
                                          simulations=simulations)

    jumbled_CI = norm.interval(0.95, loc=np.mean(jumbled_accuracies),
                               scale=np.std(jumbled_accuracies, ddof=1))

    ci_significance = np.mean(mapped_accuracies) > jumbled_CI[1]

    # Returns mapped accuracies again (kindda dumb)

    # Two-tailed t_test with Welch's correction
    t_value, p_two_tailed = ttest_ind(mapped_accuracies, jumbled_accuracies, equal_var=False)

    cohen_d = \
        (np.mean(mapped_accuracies) - np.mean(jumbled_accuracies)) / \
        np.sqrt((np.std(mapped_accuracies, ddof=1) ** 2 + np.std(jumbled_accuracies, ddof=1) ** 2) / 2)

    # Getting the one-tailed p value from a two-tailed test:
    # If T is in the hypothesis direction (t > 0 in this case): one-tailed p is half of the two-tailed
    # Elif T is in the contrary direction: one-tailed p is 1 minus half of the two-tailed
    if t_value >= 0:
        p_one_tailed = p_two_tailed / 2
    else:
        p_one_tailed = 1 - p_two_tailed / 2

    return t_value, p_one_tailed, cohen_d, ci_significance, jumbled_accuracies, mapped_accuracies


def write_data_to_csv(matrix, file_name, method, stim_list=('STIM49', 'STIM50', 'STIM51', 'STIM52'),
                      significance=None, t_value=None, p_value=None, cohen_d=None, best_sigma=None, best_bin=None):
    import pandas as pd
    import csv

    df = pd.DataFrame(matrix, columns=stim_list,
                      index=stim_list)

    with open(str(file_name + '.csv'), 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows([[file_name],
                          ['Analysis method: ' + method],
                          ['Best sigma = ' + str(best_sigma)],
                          ['Best bin = ' + str(best_bin)],
                          ['Mean > 95% value of jumbled = ' + str(significance)],
                          ['One tailed T-test statistic (> jumbled spikes) = ' + str(t_value)],
                          ['P value = ' + str(p_value)],
                          ['Cohen\'s d = ' + str(cohen_d)]])
        df.to_csv(file)


def run_classification(memory_name,
                       method,
                       sampling_rate,
                       pre_stimulus_time, post_stimulus_time,
                       pre_stimulus_raster, post_stimulus_raster,
                       brain_area,
                       write_master_sheet,
                       master_sheet_name,
                       unit_name,
                       number_of_simulations,
                       count_bin_set=(2,),
                       master_sheet_columns=(),
                       key_name=None,
                       dtw_radius=None,
                       downsample_q=None,
                       stim_names=None,
                       sigma_set=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                       save_plot=False,
                       save_rasters=False,
                       save_histograms=False,
                       write_to_csv=False, file_name=None, pdf_handle=None):
    """
    :param memory_name:
    :param method:
    :param sampling_rate:
    :param pre_stimulus_time:
    :param post_stimulus_time:
    :param pre_stimulus_raster:
    :param post_stimulus_raster:
    :param brain_area:
    :param write_master_sheet:
    :param master_sheet_name:
    :param unit_name:
    :param number_of_simulations:
    :param trial_name:
    :param key_name:
    :param dtw_radius:
    :param stim_names:
    :param sigma_set:
    :param save_plot:
    :param save_rasters:
    :param save_histograms:
    :param write_to_csv:
    :param file_name:
    :return:
    """
    import time as t
    import csv
    from datetime import date
    import multiprocessing as mp

    date_today = str(date.today())
    date_today = date_today[:4] + date_today[5:7] + date_today[8:10]  # Taking hyphens out
    logging.basicConfig(format='%(asctime)s %(processName)s %(message)s',
                        filename=master_sheet_name + "_" + mp.current_process().name + '_' + date_today + '.log',
                        level=logging.INFO)

    if key_name is None:
        key_name = memory_name

    def key_with_max_val(dictionary):
        """
        a) create a list of the dict's keys and values;
        b) return the key with the max value
        """
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(max(v))]

    best_bin = None
    best_sigma = None

    if method in ('rcorr', 'cross_corr', 'dtw'):
        logging.info('Initializing spike timing classification for ' + memory_name + '...')

        t0 = t.time()

        peristim_dict = psd.load_rasters(memory_name, key_name=key_name, sampling_rate=sampling_rate,
                                         pre_stimulus_time=pre_stimulus_time,
                                         post_stimulus_time=post_stimulus_time)
        number_of_stimuli = len(peristim_dict.keys())

        temp_matrix_dict = {}
        temp_mapped_accuracies_dict = {}
        mean_of_accuracies_dict = {}
        for current_sigma in sigma_set:
            peristim_dict = psd.spike_timing_process_data(peristim_dict, sampling_rate=sampling_rate,
                                                          sigma=current_sigma)

            temp_matrix_dict[current_sigma], temp_mapped_accuracies_dict[current_sigma] = \
                stim_classify(peristim_dict,
                              method=method,
                              dtw_radius=dtw_radius,
                              downsample_q=downsample_q,
                              trial_shuffling=True,
                              simulations=number_of_simulations,
                              brain_area=brain_area)
            # Sum of the sums of the accuracies for getting an accuracy measure
            if brain_area is 'NCM':
                mean_of_accuracies_dict[current_sigma] = np.mean(np.diag(temp_matrix_dict[current_sigma]))
            else:
                mean_of_accuracies_dict[current_sigma] = temp_matrix_dict[current_sigma][0, 0]

        best_sigma = key_with_max_val(mean_of_accuracies_dict)
        accuracy_matrix = temp_matrix_dict[best_sigma]
        mapped_accuracies = temp_mapped_accuracies_dict[best_sigma]  # For trial shuffling
        # Alter peristim_dict to become a gaussian using the best sigma
        peristim_dict = psd.spike_timing_process_data(peristim_dict, sampling_rate=sampling_rate,
                                                      sigma=best_sigma)

        plot_performance(temp_mapped_accuracies_dict, number_of_stimuli=number_of_stimuli, method=method,
                         title=file_name + " " + method)
        pdf_handle.savefig()
        plt.close()

        logging.info('Classification completed after ' + str(t.time() - t0) + ' seconds.')

    else:
        logging.info('Initializing spike count classification for ' + memory_name + '...')

        t0 = t.time()
        temp_matrix_dict = {}
        temp_mapped_accuracies_dict = {}  # For trial shuffling
        mean_of_accuracies_dict = {}

        peristim_dict = psd.load_rasters(memory_name, key_name=key_name, sampling_rate=sampling_rate,
                                         pre_stimulus_time=pre_stimulus_time,
                                         post_stimulus_time=post_stimulus_time)
        number_of_stimuli = len(peristim_dict.keys())

        for current_bin in count_bin_set:
            temp_matrix_dict[current_bin], temp_mapped_accuracies_dict[current_bin] = \
                stim_classify(peristim_dict,
                              method=method,
                              count_bin=current_bin,
                              sampling_rate=sampling_rate,
                              dtw_radius=dtw_radius,
                              downsample_q=downsample_q,
                              trial_shuffling=True,
                              simulations=number_of_simulations,
                              brain_area=brain_area)

            # Sum of the sums of the accuracies for getting an accuracy measure
            if brain_area is 'NCM':
                mean_of_accuracies_dict[current_bin] = np.mean(np.diag(temp_matrix_dict[current_bin]))
            else:
                mean_of_accuracies_dict[current_bin] = temp_matrix_dict[current_bin][0, 0]

        best_bin = key_with_max_val(mean_of_accuracies_dict)
        accuracy_matrix = temp_matrix_dict[best_bin]
        mapped_accuracies = temp_mapped_accuracies_dict[best_bin]  # For trial shuffling

        f = plot_performance(temp_mapped_accuracies_dict, number_of_stimuli=number_of_stimuli, method=method,
                             title=file_name + " " + method)
        pdf_handle.savefig()
        plt.close(f)

        logging.info('Classification completed after ' + str(t.time() - t0) + ' seconds.')

    logging.info('Initializing trial shuffling...')
    t0 = t.time()
    # Trial shuffling

    t_value, p_one_tailed, cohen_d, ci_significance, jumbled, mapped = \
        trial_shuffling_approach(peristim_dict, mapped_accuracies, count_bin=best_bin,
                                 downsample_q=downsample_q, method=method, brain_area=brain_area,
                                 simulations=number_of_simulations)

    logging.info('TSA completed after ' + str(t.time() - t0) + ' seconds.')
    # Plots

    if file_name is None:
        file_name = memory_name

    if save_histograms is True:
        f = plot_trial_shuffling_histogram(jumbled, mapped, title=file_name + " " + method)
        pdf_handle.savefig()
        plt.close(f)

    if stim_names is None:
        number_of_stimuli = len(accuracy_matrix[0])
        stim_names = ['STIM' + str(num) for num in np.arange(1, number_of_stimuli + 1)]

    if save_plot is True:
        logging.info('Plotting confusion matrices...')
        f = plot_cm(accuracy_matrix, stim_names, title=file_name + " " + method)
        pdf_handle.savefig()
        plt.close(f)

    if save_rasters is True:
        logging.info('Plotting rasters...')
        f = plot_rasters(memory_name, title=file_name + " " + method, key_name=key_name, sampling_rate=sampling_rate,
                         pre_stimulus_raster=pre_stimulus_raster, post_stimulus_raster=post_stimulus_raster,
                         post_stimulus_time=post_stimulus_time, method=method, downsample_q=downsample_q,
                         sigma=best_sigma, stim_names=stim_names)
        pdf_handle.savefig()
        plt.close(f)

    # For the purposes of the write_to_csv function
    if method is 'count':
        best_sigma = None

    if write_to_csv is True:
        logging.info('Writing to CSV...')
        write_data_to_csv(accuracy_matrix, file_name=file_name, method=method, stim_list=stim_names,
                          t_value=t_value, p_value=p_one_tailed, cohen_d=cohen_d,
                          significance=ci_significance,
                          best_sigma=best_sigma, best_bin=best_bin)

    # Headings:
    # ['Unit'] + ['Hemisphere'] + ['Method'] + ['Trial'] + ['T.statistic'] + ['P.value'] + ['Cohen.d'] +
    # ['CI.significance'] + ['Sigma'] +
    # ['Stimulus'] + ['Accuracy']

    if write_master_sheet:
        with open(str(master_sheet_name + '_accuracies.csv'), 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i, stimulus in enumerate(stim_names):
                writer.writerow(
                    [unit_name] + [method] +
                    [item for item in master_sheet_columns] +
                    [stimulus] + [accuracy_matrix[i, i]])

            with open(str(master_sheet_name + '_output.csv'), 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(
                    [unit_name] + [method] +
                    [item for item in master_sheet_columns] +
                    [t_value] + [p_one_tailed] + [cohen_d] +
                    [ci_significance] + [best_sigma] + [best_bin])
    # if write_master_sheet:
    #     if hemisphere is not None:
    #         with open(str(master_sheet_name + '_accuracies.csv'), 'a', newline='') as file:
    #             writer = csv.writer(file, delimiter=',')
    #             for i, stimulus in enumerate(stim_names):
    #                 writer.writerow(
    #                     [unit_name] + [hemisphere] + [method] + [trial_name] + [stimulus] + [accuracy_matrix[i, i]])
    #
    #         with open(str(master_sheet_name + '_output.csv'), 'a', newline='') as file:
    #             writer = csv.writer(file, delimiter=',')
    #             writer.writerow(
    #                 [unit_name] + [hemisphere] + [method] + [trial_name] + [t_value] + [p_one_tailed] + [cohen_d] +
    #                 [ci_significance] + [best_sigma] + [best_bin])
    #     else:
    #         with open(str(master_sheet_name + '_accuracies.csv'), 'a', newline='') as file:
    #             writer = csv.writer(file, delimiter=',')
    #             for i, stimulus in enumerate(stim_names):
    #                 writer.writerow(
    #                     [unit_name] + [method] + [trial_name] + [stimulus] + [accuracy_matrix[i, i]])
    #
    #         with open(str(master_sheet_name + '_output.csv'), 'a', newline='') as file:
    #             writer = csv.writer(file, delimiter=',')
    #             writer.writerow(
    #                 [unit_name] + [method] + [trial_name] + [t_value] + [p_one_tailed] + [cohen_d] +
    #                 [ci_significance] + [best_sigma] + [best_bin])




def plot_trial_shuffling_histogram(jumbled_data, mapped_data, method=None, title=None):
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(111)
    # Transform accuracy measures to percentages
    jumbled_data_transformed = np.array(jumbled_data) * 100
    mapped_data_transformed = np.array(mapped_data) * 100

    xs = np.linspace(0, 100, 2000)

    jumbled_density = gaussian_kde(jumbled_data_transformed)
    jumbled_density.covariance_factor = lambda: .2
    jumbled_density._compute_covariance()
    jumbled_CI_x = (norm.interval(0.95, loc=np.mean(jumbled_data_transformed),
                                  scale=np.std(jumbled_data_transformed, ddof=1))[1])

    shuffled_plot, = ax.plot(xs, jumbled_density(xs), label='Shuffled', color='k')
    ax.fill_between(xs, 0, jumbled_density(xs), facecolor='k', alpha=0.3)

    CI_line = ax.axvline(jumbled_CI_x, color='k', lw=2, ls='--')

    mapped_mean = np.mean(mapped_data_transformed)
    if np.var(mapped_data_transformed) != 0:
        mapped_density = gaussian_kde(mapped_data_transformed)
        mapped_density.covariance_factor = lambda: .2
        mapped_density._compute_covariance()
        mapped_plot, = ax.plot(xs, mapped_density(xs), label='Original', color='b')
        ax.fill_between(xs, 0, mapped_density(xs), facecolor='b', alpha=0.3)
    else:
        mapped_plot, = ax.plot(2000, 0, label='Original', color='b')  # invisible dot

    mean_line = ax.axvline(mapped_mean, color='b', lw=2, ls=':')

    ax.set_xlim([0, 100])

    f.legend([shuffled_plot, CI_line, mapped_plot, mean_line],
             ["Shuffled", "Shuffled 95% CI", "Original", "Original mean"], frameon=False)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Frequency of values")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if title is not None:
        f.suptitle(title)

    plt.tight_layout()

    return f


def plot_rasters(memory_name, sampling_rate, pre_stimulus_raster, post_stimulus_raster, post_stimulus_time, title,
                 method, downsample_q=None, key_name=None, sigma=None, stim_names=None, pdf_handle=None):
    """
    :param memory_name:
    :param sampling_rate:
    :param pre_stimulus_raster:
    :param post_stimulus_raster:
    :param post_stimulus_time:
    :param key_name:
    :param sigma:
    :param stim_names:
    :return:
    """

    rasters_dict = psd.load_rasters(memory_name, sampling_rate, pre_stimulus_raster,
                                    post_stimulus_raster,
                                    key_name=key_name)

    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    number_of_stimuli = len(rasters_dict.keys())

    if sigma is not None:
        rasters_dict = psd.spike_timing_process_data(
            rasters_dict, sampling_rate=sampling_rate, sigma=sigma)

    # if method is 'count':
    #     # Unified rasterplot attempt
    #     f, axarr = plt.subplots(1, number_of_stimuli, sharex='col', sharey='row')
    #     for plot_col_idx, stim_name in enumerate(rasters_dict.keys()):
    #         raster_to_plot = [psd.binary_to_time(raster, sampling_rate) for raster in rasters_dict[stim_name]]
    #         line_offsets = np.flip(np.arange(0, len(raster_to_plot)), 0)
    #         colors = np.repeat('black', len(raster_to_plot))
    #         axarr[plot_col_idx].eventplot(raster_to_plot, colors=colors, lineoffsets=line_offsets,
    #                                       linelengths=1, linewidths=0.1, alpha=0.8)
    #         axarr[plot_col_idx].axvspan(xmin=pre_stimulus_raster, xmax=pre_stimulus_raster + post_stimulus_time, color='black', alpha=0.2)
    #
    #         axarr[plot_col_idx].axis('off')
    #
    #     for idx, stim_name in enumerate(stim_names):
    #         axarr[idx].set_title(stim_name)
    #     f.suptitle(title)
    # else:
    f, axarr = plt.subplots(number_of_stimulus_repetitions, number_of_stimuli, sharex='col', sharey='row')
    f.set_size_inches((24, 13.3))

    # Populate with data
    for plot_row_idx in np.arange(0, number_of_stimulus_repetitions):
        for plot_col_idx, stim_name in enumerate(rasters_dict.keys()):
            if method is not 'count' and downsample_q is not None:
                train_to_plot = resample(rasters_dict[stim_name][plot_row_idx],
                                         len(rasters_dict[stim_name][plot_row_idx]) // downsample_q)
            else:
                train_to_plot = rasters_dict[stim_name][plot_row_idx]

            if method is not 'count':
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot,
                    color='black')
            else:
                train_to_plot[train_to_plot == 0.0] = np.nan
                axarr[plot_row_idx, plot_col_idx].plot(
                    np.linspace(-pre_stimulus_raster, post_stimulus_raster, train_to_plot.size),
                    train_to_plot, "|",
                    markersize=mpl.rcParams['font.size'],  # Change from default on purpose. 1.5 is the scale
                    markeredgewidth=mpl.rcParams['font.size'] / 100,
                    color='black', alpha=0.8)
                # axarr[plot_row_idx, plot_col_idx].set_ylim([0.9, 1.1])

            axarr[plot_row_idx, plot_col_idx].axvspan(xmin=0, xmax=post_stimulus_time, color='black', alpha=0.2)

            axarr[plot_row_idx, plot_col_idx].axis('off')

    for idx, stim_name in enumerate(stim_names):
        axarr[0, idx].set_title(stim_name)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[number_of_stimulus_repetitions - 1, :]])
    f.suptitle(title)

    plt.subplots_adjust(hspace=0)

    if pdf_handle is not None:
        pdf_handle.savefig()
        plt.close()

    return f


def plot_cm(accuracy_matrix, stim_names, title):
    import matplotlib.ticker as ticker
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(1, 1, 1)
    cax = ax.matshow(accuracy_matrix * 100, interpolation='nearest', vmin=0, vmax=100, cmap='jet')
    ax.set_xticklabels([''] + stim_names)
    ax.set_yticklabels([''] + stim_names)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    f.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return f


def plot_performance(mapped_accuracies_dict, method, number_of_stimuli, title=None):
    f = plt.figure()
    f.set_size_inches((24, 13.3))
    ax = f.add_subplot(111)

    if method is 'rcorr':
        color = 'navy'
    elif method is 'cross_corr':
        color = 'forestgreen'
    elif method is 'dtw':
        color = 'darkmagenta'
    else:
        color = 'firebrick'

    temp_items_list = sorted(list(mapped_accuracies_dict.items()))
    x_values = [key_value[0] for key_value in temp_items_list]
    y_values = [key_value[1] for key_value in temp_items_list]

    # Mean and 95% CI of each sigma/bin
    y_means = np.mean(y_values, axis=1) * 100
    y_cis = norm.interval(0.95, loc=np.mean(y_values, axis=1),
                          scale=np.std(y_values, axis=1, ddof=1))
    y_cis_range = np.subtract(y_cis[1], y_cis[0]) * 100

    cax = ax.errorbar(x_values, y_means, yerr=y_cis_range, fmt='-o', color=color, clip_on=False)
    for b in cax[1]:
        b.set_clip_on(False)

    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 20))  # outward by 10 points
                spine.set_smart_bounds(True)
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    random_prob_line = 100 / number_of_stimuli
    ax.axhline(random_prob_line, color='k', lw='2', ls='--', alpha=0.2)

    ax.set_xscale("log", basex=2)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # ax.set_xlim([ax.get_xlim()[0]-1, ax.get_xlim()[1]+1])

    # xticklabels = [tup[0] for tup in temp_items_list]
    # PERFORMANCE_AX.set_xticklabels(xticklabels)
    #
    # ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
    # ax.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax.set_xlabel('Bin/sigma size')
    ax.set_ylabel('Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    adjust_spines(ax, ['left', 'bottom'])

    plt.title(title)

    # f.show()
    plt.tight_layout()

    return f


def mutual_information(rasters_dict,
                       bin,
                       sampling_rate=0.001):
    number_of_stimuli = len(rasters_dict.keys())
    # print("Stim classify...")
    # Determining the smallest number of stimulus repetitions
    number_of_stimulus_repetitions = 1000  # Arbitrary large number
    for key in rasters_dict.keys():
        curr_stimulus_repetitions = len(rasters_dict[key])
        if curr_stimulus_repetitions < number_of_stimulus_repetitions:
            number_of_stimulus_repetitions = curr_stimulus_repetitions

    rasters_dict = dict((key_train[0], key_train[1][0:number_of_stimulus_repetitions])
                        for key_train in rasters_dict.items())

    rasters_dict = dict((key_train[0], [psd.bin_train(train, sampling_rate, bin)
                                        for train in key_train[1]])
                        for key_train in rasters_dict.items())

    def test_template(test_train, remaining_trains_dict):
        def min_indices(lst):
            """
            This function returns the indices of all the minima in the list
            """
            result = []
            offset = -1
            while True:
                try:
                    offset = lst.index(min(lst), offset + 1)
                except ValueError:
                    return result
                result.append(offset)

        min_distances_per_stim = list()
        for key in sorted(list(remaining_trains_dict.keys())):
            temp_distances_per_stim = list()
            for remaining_train in remaining_trains_dict[key]:
                # vector dot-subtraction (element-by-element, aka bin-by-bin)
                #   then square all values (squared distances) then sum them
                temp_distances_per_stim.append(np.sqrt(np.sum(np.power(np.subtract(test_train, remaining_train), 2))))
            min_distances_per_stim.append(temp_distances_per_stim[min_indices(temp_distances_per_stim)[0]])

        index_min = min_indices(min_distances_per_stim)
        # If there are ties, choose randomly
        if len(index_min) > 1:
            index_min = choice(index_min)
        return index_min

    current_matrix = np.zeros([number_of_stimuli, number_of_stimuli])
    actual_stim_index = 0
    for key in sorted(list(rasters_dict.keys())):

        for dummy_idx, train in enumerate(rasters_dict[key]):
            # Creates copy of raster dictionary to pop trains from it
            temp_raster_dict = deepcopy(rasters_dict)
            del temp_raster_dict[key][dummy_idx]
            predicted_stim_index = test_template(train, temp_raster_dict)
            current_matrix[actual_stim_index, predicted_stim_index] += 1
        actual_stim_index += 1

    current_matrix /= (number_of_stimulus_repetitions - 1.)

    # My take on MI. Need to check if it's correct:
        # MI = sum of p(r, s) * log2 {p(r, s)/[p(r)*p(s)]}, where:
            # p(r, s) = p(r|s)*p(s): classification accuracy for a given stimulus s multiplied by p(s)
            # p(r) = 1/stimulus_repetitions (probability of classifying response as any given stimulus. Unsure about that...
            # p(s) = 1/number_of_stimuli
            # So it comes down to: sum of ( accuracy*p(s) * log2 {accuracy / p(r)} )
    mi = 0
    prob_stim = 1./number_of_stimuli
    for index in range(0, number_of_stimuli):
        curr_accuracy = current_matrix[index, index]
        curr_result = curr_accuracy * prob_stim * np.log2(curr_accuracy*number_of_stimulus_repetitions)
        if np.isnan(curr_result):
            curr_result = 0
        mi += curr_result

    return current_matrix, mi


def run_mi(memory_name,
           sampling_rate,
           pre_stimulus_time, post_stimulus_time,
           bin_set=(2,),
           key_name=None):
    import time as t
    import csv
    def key_with_max_val(dictionary):
        """
        a) create a list of the dict's keys and values;
        b) return the key with the max value
        """
        v = list(dictionary.values())
        k = list(dictionary.keys())
        return k[v.index(max(v))]
    logging.info('Initializing spike count classification for ' + memory_name + '...')

    t0 = t.time()
    matrices_dicts = {}
    mi_dicts = {}  # For trial shuffling

    peristim_dict = psd.load_rasters(memory_name, key_name=key_name, sampling_rate=sampling_rate,
                                     pre_stimulus_time=pre_stimulus_time,
                                     post_stimulus_time=post_stimulus_time)
    for current_bin in bin_set:
        matrices_dicts[current_bin], mi_dicts[current_bin] = \
            mutual_information(peristim_dict,
                               bin=current_bin,
                               sampling_rate=sampling_rate)

    best_bin = key_with_max_val(mi_dicts)
    return best_bin, mi_dicts[best_bin]
