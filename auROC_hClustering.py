import numpy as np
import json
import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.signal import resample
from astropy.convolution import convolve_fft, Gaussian1DKernel, Box1DKernel
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, MeanShift, KMeans
from glob import glob
import pandas as pd
import seaborn as sns
sns.set_theme(color_codes=True, palette="Set2")
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import pairwise_distances
from os.path import sep
import platform

from rpy2.robjects.packages import importr, data

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max + 1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, k_max + 1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            # km = KMeans(k)
            km = AgglomerativeClustering(distance_threshold=None, n_clusters=k, linkage='ward', compute_distances=True)
            km.fit(randomReference)

            # refDisp = km.inertia_
            refDisp = np.sum(km.distances_**2)
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        # km = AgglomerativeClustering(distance_threshold=None, n_clusters=k, linkage='ward', compute_distances=True)
        km.fit(data)

        origDisp = km.inertia_
        # origDisp = np.sum(km.distances_**2)

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

#
# def zscore_response_curve(hist, edges, pre_stimulus_baseline_start, pre_stimulus_baseline_end, window=0.1):
#     # Baseline from -2 to -1 s
#     baseline_points_mask = (edges >= -pre_stimulus_baseline_start) & (edges < -pre_stimulus_baseline_end)
#     baseline_hist = hist[baseline_points_mask[:-1]]
#
#     baseline_mean = np.nanmean(baseline_hist)
#     baseline_std = np.nanstd(baseline_hist, ddof=1)
#
#     # For every bin during response
#     zscore_curve = []
#     for start_bin in np.arange(edges[0], edges[-1], window):
#         cur_points_mask = (edges >= start_bin) & (edges < start_bin + window)
#         cur_hist_values = hist[cur_points_mask[:-1]]
#
#         cur_mean = np.mean(cur_hist_values)
#
#         zscore_curve.append((cur_mean - baseline_mean)/baseline_std)
#         # # For debugging
#         # mpl.use('TkAgg')
#         # plt.figure()
#         # plt.plot(false_positive, true_positive)
#         # plt.show()
#
#     return zscore_curve


BASELINE_DURATION_FOR_FR = 1  # in seconds; for firing rate calculation to non-AM trials
STIM_DURATION_FOR_FR = 1  # in seconds; for firing rate calculation to AM trials
PRETRIAL_DURATION_FOR_SPIKETIMES = 2  # in seconds; for grabbing spiketimes around AM trials
POSTTRIAL_DURATION_FOR_SPIKETIMES = 5  # in seconds; for grabbing spiketimes around AM trials



OUTPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output'])
INPUT_FOLDER = '.' + sep + sep.join(['Data', 'Output', 'JSON files'])
# Load data and calculate auROCs based on trial responses aligned to SpoutOff triggering a hit
all_json = glob(INPUT_FOLDER + sep + '*json')

BINSIZE = 0.1
CLUSTERING_TIME_START = 0
CLUSTERING_TIME_END = 2
col_name = ['Hit_auroc', 'SpoutOff_hits_auroc', 'Miss_auroc', 'FA_auroc', 'allSpoutOnset_auroc', 'allSpoutOffset_auroc',
            'AMTrial_auroc', 'allSpoutOffset_auroc']

# CLUSTERING_TIME_START = -1
# CLUSTERING_TIME_END = 2
# col_name = ['allSpoutOnset_auroc']

# CLUSTERING_TIME_START = 0
# CLUSTERING_TIME_END = 2
# col_name = ['Hit_psth', 'SpoutOff_hits_psth', 'Miss_psth', 'FA_psth', 'allSpoutOnset_psth', 'allSpoutOffset_psth',
#             'AMTrial_psth', 'allSpoutOnset_psth', 'allSpoutOffset_psth']

# CLUSTERING_TIME_START = -1
# CLUSTERING_TIME_END = 2
# col_name = ['allSpoutOnset_psth']

for cur_col in col_name:
    unit_list = []
    auroc_list = []
    for file_name in all_json:
        with open(file_name, 'r') as json_file:
            cur_dict = json.load(json_file)
        # Grab Active data
        session_names = list(cur_dict['Session'].keys())
        try:
            active_session_name = [s for s in session_names if ("Aversive" in s) or ("Active" in s)][0]
        except IndexError:
            continue



        active_data = cur_dict['Session'][active_session_name]

        # if variable == 'SpoutOff_hits_auroc':
        #     # Last second has some "dark spots" because
        #     # the trial-alignment ran up to 4s after trial onset, not after spoutOff triggering hit
        #     auroc_curve = np.array(active_data[variable]
        #                            [:int((3 + PRETRIAL_DURATION_FOR_SPIKETIMES) / AUROC_BINSIZE)])
        # else:

        response_curve = np.array(active_data[cur_col])

        # Exclude units without responses (no FA trials for example)
        if len(response_curve) == 0:
            continue

        # Z-score firing rate
        if 'psth' in cur_col:
            boxcar_points = 10
            # response_curve = convolve_fft(response_curve, Box1DKernel(boxcar_points))
            response_curve = resample(response_curve, len(response_curve) // boxcar_points)

            relevant_indices = np.arange((-PRETRIAL_DURATION_FOR_SPIKETIMES) /
                                         BINSIZE, (-PRETRIAL_DURATION_FOR_SPIKETIMES + BASELINE_DURATION_FOR_FR) / BINSIZE,dtype=np.int32)
            response_curve_baseline_mean = np.mean(response_curve[relevant_indices])
            response_curve_baseline_sd = np.std(response_curve[relevant_indices], ddof=1)
            if response_curve_baseline_sd == 0:
                continue
            response_curve = (response_curve - response_curve_baseline_mean)/response_curve_baseline_sd



        auroc_list.append(response_curve)
        unit_list.append(cur_dict['Unit'])


    # setting distance_threshold=0 ensures we compute the full tree.

    plot_list = np.array(auroc_list)




    # Rescale and remove nan if zscore
    # plot_list = [ (x - np.min(x)) / (np.max(x) - np.min(x)) for x in zscore_miss_list if ~np.isnan(np.sum(x))]
    # plot_list = [ x for x in zscore_miss_list if ~np.isnan(np.sum(x))]
    # model = AgglomerativeClustering(n_clusters=None, linkage='ward',
    #                                 compute_distances=True, compute_full_tree=True, distance_threshold=0)
    # model = model.fit(plot_list)
    # model = model.fit(plot_list)
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model, truncate_mode='level', p=1)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    # Seaborn


    relevant_indices = np.arange((CLUSTERING_TIME_START + PRETRIAL_DURATION_FOR_SPIKETIMES) /
                                 BINSIZE, (CLUSTERING_TIME_END + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE)

    relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]] for cur_auroc in plot_list])
    # relevant_snippet = [cur_auroc for cur_auroc in auroc_list]


    # link_type = ['single', 'complete', 'average', 'ward']
    # cophen_corrs = []
    # for i in link_type:
    #     clust_dist = linkage(relevant_snippet, method=i, metric='euclidean')
    #
    #     # find the cophenetic correlation by comparing the clustering distances
    #     # and the euclidean distances
    #     cophen_corr, cophen_array = cophenet(clust_dist, pdist(relevant_snippet))
    #     cophen_corrs.append(cophen_corr)
    #
    #     print('cophenetic correlation of ' + i + ' method: ', cophen_corr)
    #
    # # Choose best link type based on cophenet
    # row_link = linkage(relevant_snippet, method=link_type[cophen_corrs.index(np.max(cophen_corrs))], metric='euclidean')

    row_link = linkage(relevant_snippet, method='ward', metric='euclidean', optimal_ordering=False)

    last = row_link[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    # k = 5
    print("clusters:", k)
    clusters = fcluster(row_link, k, criterion='maxclust')

    # Color cluster separation
    unique_clusters = np.unique(clusters.flatten())
    palette = sns.husl_palette(len(unique_clusters))

    color_dict = dict()
    for dummy_idx, unique_cluster in enumerate(unique_clusters):
        color_dict.update({unique_cluster: palette[dummy_idx]})

    cluster_df = pd.DataFrame({"Cluster_id": clusters})

    row_colors = cluster_df.Cluster_id.map(color_dict)

    with PdfPages(sep.join([OUTPUT_FOLDER, cur_col + '_HClustering.pdf'])) as pdf:
        plt.figure()

        # If you desire to smooth the heatmap
        # g = sns.clustermap([convolve_fft(cell_z, Gaussian1DKernel(stddev=3), preserve_nan=True) for cell_z in plot_list],
        #                    row_cluster=True, col_cluster=False, row_linkage=row_link, vmin=-5, vmax=5,
        #                    row_colors=row_colors.values)

        g = sns.clustermap(plot_list, row_cluster=True, col_cluster=False, row_linkage=row_link, row_colors=row_colors.values)

        g.ax_heatmap.set_xticklabels( [np.round(float(a.get_text()) * BINSIZE - PRETRIAL_DURATION_FOR_SPIKETIMES, 1)
                                       for a in g.ax_heatmap.get_xticklabels()] , size='xx-small')
        g.ax_heatmap.set_xlabel('Time relative to event (s)')
        g.ax_heatmap.set_ylabel('Unit # / clustering')
        g.ax_heatmap.set_title(cur_col)
        g.ax_heatmap.axvline(x=(CLUSTERING_TIME_START + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE, color='lightcyan', linestyle='--')
        g.ax_heatmap.axvline(x=(CLUSTERING_TIME_END + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE, color='lightcyan', linestyle='--')

        if cur_col == 'SpoutOff_hits_auroc':
            g.ax_heatmap.set_xlim([0, (POSTTRIAL_DURATION_FOR_SPIKETIMES + PRETRIAL_DURATION_FOR_SPIKETIMES - 1) / BINSIZE])

        # plt.show()
        pdf.savefig()
        plt.close()

    # Plot average response by cluster group
    # Transform responses and cluster id into a dataframe
    x_axis = np.arange(0, np.size(plot_list, 1)) * BINSIZE - PRETRIAL_DURATION_FOR_SPIKETIMES
    df = pd.DataFrame({'Unit': np.repeat(unit_list, np.size(plot_list, 1)),
                       'Time_s': np.tile(x_axis, np.size(plot_list, 0)),
                       'auROC': plot_list.flatten(),
                       'Cluster': np.repeat(clusters, np.size(plot_list, 1))})
    df = df.convert_dtypes()
    df = df.astype({'Cluster': 'category'})

    with PdfPages(sep.join([OUTPUT_FOLDER, cur_col + '_meanResponse.pdf'])) as pdf:
        fig, ax = plt.subplots()
        sns.set_style("ticks")
        # g = sns.relplot(data=df, x="Sample", y="auROC", hue="Cluster", palette='Set2',
        #             units='Unit', kind='line', estimator=None, alpha=0.2, ax=ax)
        g = sns.relplot(data=df, x="Time_s", y="auROC", hue="Cluster", kind='line', ax=ax,
                        palette=sns.husl_palette(len(unique_clusters)))  # 95% CI is the default
        # g.ax.set_xticks(np.arange(0, np.size(plot_list, 1)+1, 5))  # <--- set the ticks first
        # g.ax.set_xticklabels(g.ax.get_xticks() * AUROC_BINSIZE - PRETRIAL_DURATION_FOR_SPIKETIMES)
        g.ax.axvline(x=0, color='black',
                             linestyle='--')
        g.ax.fill_betweenx(y=[0, 1], x1=CLUSTERING_TIME_START, x2=CLUSTERING_TIME_END, facecolor='black', alpha=0.1)
        g.ax.set_xlabel('Time relative to event (s)')

        if 'auROC' in cur_col:
            g.ax.set_ylabel('auROC')
            g.ax.set_ylim([0, 1])
        elif 'psth' in cur_col:
            g.ax.set_ylabel('Spikes/s')


        if cur_col == 'SpoutOff_hits_auroc':
            g.ax.set_xlim([-PRETRIAL_DURATION_FOR_SPIKETIMES, POSTTRIAL_DURATION_FOR_SPIKETIMES - 1])

        sns.despine()
        # plt.show()
        pdf.savefig()
        plt.close()


    df.to_csv(sep.join([OUTPUT_FOLDER, cur_col + '_HClustering.csv']), index=False)



    #
    # # Check if subjects cluster together
    # color_dict=dict(zip(np.unique([unit_name[0:11] for unit_name in unit_list]), np.array(['lightcoral', 'seagreen', 'royalblue'])))
    # subject_df = pd.DataFrame({"Subject": [unit_name[0:11] for unit_name in unit_list]})
    # row_colors = subject_df.Subject.map(color_dict)
    #
    # g = sns.clustermap(plot_list, row_cluster=True, col_cluster=False, row_linkage=row_link, row_colors=row_colors.values)
    # g.ax_heatmap.set_xticklabels( [np.round(float(a.get_text()) * zscore_binsize - pre_stimulus_raster, 1) for a in g.ax_heatmap.get_xticklabels()] , size='xx-small')
    # g.ax_heatmap.set_xlabel('Time (s)')
    # plt.show()

    #
    # k_max = 50
    # gap, reference_inertia, ondata_inertia = compute_gap(AgglomerativeClustering(), relevant_snippet, k_max)
    #
    # plt.plot(range(1, k_max+1), reference_inertia,
    #          '-o', label='reference')
    # plt.plot(range(1, k_max+1), ondata_inertia,
    #          '-o', label='data')
    # plt.xlabel('k')
    # plt.ylabel('log(inertia)')
    # plt.show()
    #
    # plt.plot(range(1, k_max+1), gap, '-o')
    # plt.ylabel('gap')
    # plt.xlabel('k')
    # plt.show()