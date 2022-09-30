import numpy as np
import json
import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.signal import resample
from astropy.convolution import convolve_fft, Gaussian1DKernel, Box1DKernel
from scipy.spatial.distance import pdist
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, MeanShift
from glob import glob
import pandas as pd
import seaborn as sns
sns.set_theme(color_codes=True, palette="Set2")
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import pairwise_distances
from os.path import sep
import platform

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
numpy2ri.activate()

from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import SexpVector
rfactoextra = importr('factoextra')
rbase = importr('base')
rstats = importr('stats')
rnbclust = importr('NbClust')
rutils = importr('utils')
rfastcluster = importr('fastcluster')

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


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
            'AMTrial_auroc', 'allSpoutOnset_auroc', 'allSpoutOffset_auroc']

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

    def numpy_to_rpy2(arr):
        with localconverter(ro.default_converter + numpy2ri.converter):
            return ro.conversion.py2rpy(arr)

    # Seaborn
    relevant_indices = np.arange((CLUSTERING_TIME_START + PRETRIAL_DURATION_FOR_SPIKETIMES) /
                                 BINSIZE, (CLUSTERING_TIME_END + PRETRIAL_DURATION_FOR_SPIKETIMES) / BINSIZE)

    relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]] for cur_auroc in plot_list])

    rrelevant_snippet = numpy_to_rpy2(relevant_snippet)

    gs = rfactoextra.fviz_nbclust(rrelevant_snippet, rfactoextra.hcut, method="silhouette", nboot=500) # diss=rstats.dist(relevant_snippet, method="minkowski"))

    rutils.write_table(gs.rx2('data'), OUTPUT_FOLDER + sep + cur_col + '_clustering_gap_stat.csv', row_names = False)

    clusters = rfastcluster.hclust(rstats.dist(rrelevant_snippet), method='ward.D2')
    clusters.rx2('cluster') = rstats.cutree(clusters, k=2)


    # with localconverter(ro.default_converter + pandas2ri.converter):
    #     relevant_snippet = ro.conversion.py2rpy(relevant_snippet)







    row_link = linkage(relevant_snippet, method='ward', metric='euclidean', optimal_ordering=False)

    last = row_link[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]

    # k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    k = 5
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