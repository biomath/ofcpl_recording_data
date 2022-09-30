from os.path import sep
import platform
import json
from time import time
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import seaborn as sns

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
from matplotlib.backends.backend_pdf import PdfPages

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def optimalK(data, maxClusters=10, boot_n=10, sk_factor=1):
    """
    This function calculates the optimal number of clusters from time series
    It is optimal for time series because it uses dynamic time warping (DTW) as a distance metric
    The optimal number of clusters is derived from an implementation of the gap-statistic by Tibshirani et al., 2001

    - **parameters**, **types**, **return** and **return types**::
        :param data: time-series data
        :param maxClusters: maximum number of clusters to be evaluated
        :param boot_n: number of iterations (maximum of 50 recommended due to computing time for DTW)
        :param sk_factor: error multiplication factor for choosing best number of clusters
        :return:
            seMax_cluster: optimal number of clusters
            (mean_per_cluster, sks_per_cluster): decision parameters for plotting the gap-statistic graph

        :type data: 2D-numpy.array or list of lists containing numbers
        :type maxClusters: int
        :type boot_n: int
        :type sk_factor: int
        :rtype:
            seMax_cluster: int
            (mean_per_cluster, sks_per_cluster): tuple of numpy.arrays
    """
    gaps = np.zeros((len(range(1, maxClusters + 1)), boot_n))

    for gap_index, k in enumerate(range(1, maxClusters + 1)):
        # Cluster data set to get inertia (aka MSE(within)/MSE(between) clusters)
        km = TimeSeriesKMeans(n_clusters=k, metric='dtw', init='k-means++')
        km.fit(data)
        data_inertia = km.inertia_

        for boot_index in range(0, boot_n):
            # Create new random reference set and cluster to get inertia
            randomReference = np.random.random_sample(size=data.shape)

            km = TimeSeriesKMeans(n_clusters=k, metric='dtw', init='k-means++',)
            km.fit(randomReference)

            reference_inertia = km.inertia_

            # Calculate gap statistic
            gap = np.log(reference_inertia) - np.log(data_inertia)

            # Assign this loop's gap statistic to gaps
            gaps[gap_index, boot_index] = gap

    mean_per_cluster = np.mean(gaps, axis=1)

    # Tibshirani et al., 2001's error: s(k)_factor * s(k)
    sks_per_cluster = sk_factor * ( np.sqrt(1 + 1/boot_n) * np.std(gaps, axis=1, ddof=0) )
    tibs_score = [cur_mean - cur_sk for cur_mean, cur_sk in zip(mean_per_cluster, sks_per_cluster)]

    first_seMax = -1000  # arbitrarily large
    seMax_found = False
    # Loop through clusters in order until criterion is met
    # “Tibshirani et al (2001) proposed: the smallest k such that f(k) ≥ f(k+1) - s{k+1}”
    for cluster_idx in np.arange(0, maxClusters-1):
        if seMax_found:
            break
        if mean_per_cluster[cluster_idx] <= tibs_score[cluster_idx + 1]:
            first_seMax = mean_per_cluster[cluster_idx]
        else:
            first_seMax = mean_per_cluster[cluster_idx]
            seMax_found = True

    seMax_cluster = np.where(mean_per_cluster == first_seMax)[0][0] + 1

    return seMax_cluster, (mean_per_cluster, sks_per_cluster)


'''
Define globals
'''

# These were parameters used in constructing the PSTHs or auROC curves; should not change if kept default
BASELINE_DURATION_FOR_FR = 1  # in seconds; for firing rate calculation to non-AM trials
STIM_DURATION_FOR_FR = 1  # in seconds; for firing rate calculation to AM trials
PRETRIAL_DURATION_FOR_SPIKETIMES = 2  # in seconds; for grabbing spiketimes around AM trials
POSTTRIAL_DURATION_FOR_SPIKETIMES = 5  # in seconds; for grabbing spiketimes around AM trials
BINSIZE = 0.1

# IO locations
OUTPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output'])
INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
# Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
all_json = glob(INPUT_FOLDER + sep + '*json')

# These are the clustering periods in relation to the event; in seconds
CLUSTERING_TIME_START = 0
CLUSTERING_TIME_END = 2

# optimalK clustering and gap-stat parameters
# Processing time estimate is ~number_of_series * MAXCLUSTERS * BOOT_N * T
# T is ~0.008 s in the CarasLab data analysis computer
MAXCLUSTERS = 10
BOOT_N = 50
SK_FACTOR = 2

# Sessions you want to cluster; keep in mind passive sessions do not contain true trial outcomes or spout events and
# should only be clustered using all AMTrials or by AMdepth
which_session = ['active']  # pre, active, post, post1h

# Events you want to cluster
# col_name = ['Hit_auroc', 'Miss_auroc', 'FA_auroc', 'AMTrial_auroc',  # Trial stuff
#             'SpoutOff_hits_auroc', 'allSpoutOnset_auroc', 'allSpoutOffset_auroc']  # Spout stuff
col_name = ['AMdepth_Hit_0.25_auroc', 'AMdepth_Hit_0.35_auroc', 'AMdepth_Miss_0.25_auroc', 'AMdepth_Miss_0.35_auroc']

for session in which_session:
    for cur_col in col_name:
        print("Running clustering for metric: " + cur_col)
        t0 = time()
        unit_list = []
        auroc_list = []
        for file_name in all_json:
            # Remove SUBJ-ID-154_210520 units from these analyses because this was an extinction day
            if 'SUBJ-ID-154_210520' in file_name:
                continue

            # Open JSON
            with open(file_name, 'r') as json_file:
                cur_dict = json.load(json_file)

            # Grab  data
            session_names = list(cur_dict['Session'].keys())
            if session == 'active':
                try:
                    session_name = [s for s in session_names if ("Aversive" in s) or ("Active" in s)][0]
                except IndexError:
                    continue
            elif session == 'pre':
                try:
                    session_name = [s for s in session_names if ("Pre" in s)][0]
                except IndexError:
                    continue

            elif session == 'post':
                try:
                    session_name = [s for s in session_names if ("Post_" in s) or ("Post-" in s)][0]
                except IndexError:
                    continue
            elif session == 'post1h':
                    try:
                        session_name = [s for s in session_names if ("Post1h" in s)][0]
                    except IndexError:
                        continue
            else:
                print('Session name undefined')
                exit()


            cur_data = cur_dict['Session'][session_name]

            response_curve = np.array(cur_data[cur_col])

            # Exclude units without responses (no FA trials for example)
            if len(response_curve) == 0:
                continue

            # Option to smooth and Z-score firing rate if using non-normalized data
            # boxcar_points = 10
            # # response_curve = convolve_fft(response_curve, Box1DKernel(boxcar_points))
            # response_curve = resample(response_curve, len(response_curve) // boxcar_points)
            #
            # relevant_indices = np.arange((-PRETRIAL_DURATION_FOR_SPIKETIMES) /
            #                              BINSIZE, (-PRETRIAL_DURATION_FOR_SPIKETIMES + BASELINE_DURATION_FOR_FR) / BINSIZE,dtype=np.int32)
            # response_curve_baseline_mean = np.mean(response_curve[relevant_indices])
            # response_curve_baseline_sd = np.std(response_curve[relevant_indices], ddof=1)
            # if response_curve_baseline_sd == 0:
            #     continue
            # response_curve = (response_curve - response_curve_baseline_mean)/response_curve_baseline_sd

            auroc_list.append(response_curve)
            unit_list.append(cur_dict['Unit'])

        plot_list = np.array(auroc_list)

        # Rescale and remove nan if zscore
        # plot_list = [ (x - np.min(x)) / (np.max(x) - np.min(x)) for x in zscore_miss_list if ~np.isnan(np.sum(x))]
        # plot_list = [ x for x in zscore_miss_list if ~np.isnan(np.sum(x))]

        clustering_time_start = CLUSTERING_TIME_START

        # I opted to start clustering spout events a little earlier
        if cur_col in ('allSpoutOnset_auroc', 'allSpoutOffset_auroc'):
            clustering_time_start = -1

        relevant_indices = np.arange((clustering_time_start + PRETRIAL_DURATION_FOR_SPIKETIMES) /
                                     BINSIZE, (CLUSTERING_TIME_END + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE)

        relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]] for cur_auroc in plot_list])


        k, (gapStat_means, gapStat_sks) = optimalK(relevant_snippet, maxClusters=MAXCLUSTERS, boot_n=BOOT_N, sk_factor=SK_FACTOR)
        gapstat_df = pd.DataFrame({"Cluster_n": np.arange(1, MAXCLUSTERS + 1), "Gap_mean": gapStat_means, "Gap_sks": gapStat_sks})
        # Save gap-stat data
        gapstat_df.to_csv(sep.join([OUTPUT_FOLDER, cur_col + '_' + session + '_gapStat.csv']), index=False)

        print("clusters:", k)

        with PdfPages(sep.join([OUTPUT_FOLDER, cur_col + '_' + session + '_gapStat.pdf'])) as pdf:
            fig, ax = plt.subplots()
            ax.errorbar(np.arange(1, MAXCLUSTERS + 1), gapStat_means, gapStat_sks)
            ax.axvline(x=k, color='black', linestyle='--')
            ax.set_xlabel('Clusters')
            ax.set_ylabel('Gap-statistic')
            pdf.savefig()
            plt.close()

        # Final clustering
        row_link = TimeSeriesKMeans(n_clusters=k, metric='dtw', init='k-means++')
        row_link.fit(relevant_snippet)

        clusters = row_link.labels_ + 1

        # Color cluster separation
        unique_clusters = np.unique(clusters.flatten())
        palette = sns.husl_palette(len(unique_clusters))

        color_dict = dict()
        for dummy_idx, unique_cluster in enumerate(unique_clusters):
            color_dict.update({unique_cluster: palette[dummy_idx]})

        cluster_df = pd.DataFrame({"Cluster_id": clusters})

        row_colors = cluster_df.Cluster_id.map(color_dict)
        cluster_df['Cluster_color'] = [row_color for row_color in row_colors.values]
        cluster_df = cluster_df.sort_values(by='Cluster_id')

        with PdfPages(sep.join([OUTPUT_FOLDER, cur_col + '_' + session + '_HClustering.pdf'])) as pdf:
            plt.figure()

            # Plot with seaborn
            g = sns.clustermap(plot_list[cluster_df.index.tolist()], row_cluster=False, col_cluster=False, row_colors=row_colors[cluster_df.index.tolist()].to_numpy())
            g.ax_heatmap.set_xticklabels( [np.round(float(a.get_text()) * BINSIZE - PRETRIAL_DURATION_FOR_SPIKETIMES, 1)
                                           for a in g.ax_heatmap.get_xticklabels()] , size='xx-small')
            g.ax_heatmap.set_xlabel('Time relative to event (s)')
            g.ax_heatmap.set_ylabel('Unit # / clustering')
            g.ax_heatmap.set_title(cur_col)
            g.ax_heatmap.axvline(x=(clustering_time_start + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE, color='lightcyan', linestyle='--')
            g.ax_heatmap.axvline(x=(CLUSTERING_TIME_END + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE, color='lightcyan', linestyle='--')

            # Trim a bit of the SpoutOff_hits heatmap because the offset to get the spout off event sometimes goes beyond
            # what I used to calculate the auROC so we end up with blank spaces at the end
            if cur_col == 'SpoutOff_hits_auroc':
                g.ax_heatmap.set_xlim([0, (POSTTRIAL_DURATION_FOR_SPIKETIMES + PRETRIAL_DURATION_FOR_SPIKETIMES - 1) / BINSIZE])

            pdf.savefig()
            plt.close()

        # Plot average response by cluster group
        # Transform responses and cluster id into a dataframe first
        x_axis = np.arange(0, np.size(plot_list, 1)) * BINSIZE - PRETRIAL_DURATION_FOR_SPIKETIMES
        df = pd.DataFrame({'Unit': np.repeat(unit_list, np.size(plot_list, 1)),
                           'Time_s': np.tile(x_axis, np.size(plot_list, 0)),
                           'auROC': plot_list.flatten(),
                           'Cluster': np.repeat(clusters, np.size(plot_list, 1))})
        df = df.convert_dtypes()
        df = df.astype({'Cluster': 'category'})

        # Save data
        df.to_csv(sep.join([OUTPUT_FOLDER, cur_col + '_' + session + '_HClustering.csv']), index=False)

        # Plot average+-95%CI responses per cluster
        with PdfPages(sep.join([OUTPUT_FOLDER, cur_col + '_' + session + '_meanResponse.pdf'])) as pdf:
            fig, ax = plt.subplots()
            g = sns.relplot(data=df, x="Time_s", y="auROC", hue="Cluster", kind='line', ax=ax,
                            palette=sns.husl_palette(len(unique_clusters)))  # 95% CI is the default
            g.ax.axvline(x=0, color='black',
                                 linestyle='--')
            g.ax.fill_betweenx(y=[0, 1], x1=clustering_time_start, x2=CLUSTERING_TIME_END, facecolor='black', alpha=0.1)
            g.ax.set_xlabel('Time relative to event (s)')

            if 'auROC' in cur_col:
                g.ax.set_ylabel('auROC')
                g.ax.set_ylim([0, 1])
            elif 'psth' in cur_col:
                g.ax.set_ylabel('Spikes/s')

            # Trim a bit of the SpoutOff_hits heatmap because the offset to get the spout off event sometimes goes beyond
            # what I used to calculate the auROC so we end up with blank spaces at the end
            if cur_col == 'SpoutOff_hits_auroc':
                g.ax.set_xlim([-PRETRIAL_DURATION_FOR_SPIKETIMES, POSTTRIAL_DURATION_FOR_SPIKETIMES - 1])

            sns.despine()
            pdf.savefig()
            plt.close()

        runtime = np.round(time() - t0, 3)
        print("Runtime: " + str(runtime) + " s")