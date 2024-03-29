import json
from os import remove
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process

from get_fr_toTrials import *
from get_zscore_toTrials import *
from calculate_auROC import *
from Trash.get_fr_allSpoutTimestamps import *

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_pipeline(input_list):
    memory_path, all_json = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_200711_concat_cluster41
    split_timestamps_name = split("_*_", unit_id)  # split timestamps
    cur_date = split_timestamps_name[1]
    subject_id = split_timestamps_name[0]
    recording_type = RECORDING_TYPE_DICT[subject_id]

    # For debugging purposes
    # if unit_id != "SUBJ-ID-197_210511_concat_cluster820":
    #     continue
    # first_entry_flag = True

    # Use subj-session identifier to grab appropriate key
    # Stimulus info is in trialInfo

    # These are in alphabetical order. Must sort by date_trial or match with filev
    # Match by name for now for breakpoints

    key_paths_info = glob(KEYS_PATH + sep + subject_id + '*' +
                          cur_date + "*_trialInfo.csv")
    key_paths_spout = glob(KEYS_PATH + sep + subject_id + '*' +
                           cur_date + "*spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        # Maybe the key file wasn't found because date is in Intan format
        # Convert date to ePsych format
        cur_date = datetime.strptime(cur_date, '%y%m%d')
        cur_date = datetime.strftime(cur_date, '%y-%m-%d')
        key_paths_info = glob(KEYS_PATH + sep + subject_id + '*' +
                              cur_date + "*_trialInfo.csv")
        key_paths_spout = glob(KEYS_PATH + sep + subject_id + '*' +
                               cur_date + "*_spoutTimestamps.csv")

    if len(key_paths_info) == 0:
        print("Key not found for " + unit_id)
        return
    try:
        cur_breakpoint_file = glob(BREAKPOINT_PATH + sep +
                                   "_".join(split_timestamps_name[0:3]) + "_breakpoints.csv")[0]
        cur_breakpoint_df = read_csv(cur_breakpoint_file)
    except IndexError:
        print("Breakpoint file not found for " + unit_id + ". Assuming non-concatenated...")
        cur_breakpoint_df = pd.DataFrame()

    # If no JSON for that unit exists, create UnitData
    try:
        cur_unitData_name = all_json[
            all_json.index(OUTPUT_PATH + sep + 'JSON files' + sep + unit_id + '_unitData.json')]
        with open(cur_unitData_name, 'r') as json_file:
            cur_unitData = json.load(json_file)
    except ValueError:
        cur_unitData = {'Unit': unit_id, 'Session': {}}

    for key_path_info in key_paths_info:
        if recording_type == 'synapse':
            key_f = split(REGEX_SEP, key_path_info)[-1]
            key_f = split("_*_", key_f)[1]
        else:
            key_f = split(REGEX_SEP, key_path_info)[-1]
            key_f = split("_*_", key_f)
            key_f = '_'.join(key_f[1:4])
            # For some intan recordings, I added an extra SUBJ field before the key identifier. Patch it here
            if 'Passive' not in key_f and 'Active' not in key_f and 'Aversive' not in key_f and 'Exctinction' not in key_f:
                key_f = split(REGEX_SEP, key_path_info)[-1]
                key_f = split("_*_", key_f)
                key_f = '_'.join(key_f[2:5])

        key_path_spout_finder = [search(key_f, file_name) for file_name in key_paths_spout]
        try:
            key_path_spout_finder = [i for i, x in enumerate(key_path_spout_finder) if x is not None][0]
            key_path_spout = key_paths_spout[key_path_spout_finder]
        except IndexError:
            key_path_spout = None

        breakpoint_offset_idx = cur_breakpoint_df.index[
            cur_breakpoint_df['Session_file'].str.contains(key_f)]
        try:
            breakpoint_offset = cur_breakpoint_df.loc[
                breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]  # grab previous session's breakpoint
        except KeyError:  # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
            try:
                breakpoint_offset = cur_breakpoint_df.loc[
                    breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
            except IndexError:
                print('Something off with ' + subject_id + ', file ' + key_f)
            except KeyError:
                breakpoint_offset = 0  # first file; no breakpoint offset needed
            if recording_type == 'synapse':
                breakpoint_offset = breakpoint_offset / 24414.0625
            elif recording_type == 'intan':
                breakpoint_offset = breakpoint_offset / 30000
            else:
                print('Recording type not specified. Skipping;...')
                continue

        # Add keys if they don't exist
        try:
            if split(REGEX_SEP, key_path_info)[-1][:-4] not in cur_unitData["Session"]:
                cur_unitData["Session"].update({split(REGEX_SEP, key_path_info)[-1][:-4]: {}})
        except KeyError:
            cur_unitData["Session"].update({split(REGEX_SEP, key_path_info)[-1][:-4]: {}})

        # Flag to indicate this is the first entry to the CSV file so headers will be printed
        # Check if worker file already exists then turn flag to false
        if len(glob(OUTPUT_PATH + sep + current_process().name + "_tempfile_" + CSV_PRENAME + '*.csv')) > 0:
            first_entry_flag = False
        else:
            first_entry_flag = True

        # cur_unitData = get_fr_toTrials(memory_path,
        #                                key_path_info,
        #                                key_path_spout,
        #                                unit_name=unit_id,
        #                                csv_pre_name=current_process().name + "_tempfile_" + CSV_PRENAME,
        #                                first_cell_flag=first_entry_flag,
        #                                baseline_duration_for_fr_s=BASELINE_DURATION_FOR_FR,
        #                                stim_duration_for_fr_s=STIM_DURATION_FOR_FR,
        #                                pre_stim_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
        #                                post_stim_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES,
        #                                output_path=OUTPUT_PATH,
        #                                breakpoint_offset=breakpoint_offset,
        #                                cur_unitData=cur_unitData,
        #                                afterTrial_FR_start=AFTERTRIAL_FR_START,
        #                                afterTrial_FR_end=AFTERTRIAL_FR_END
        #                                )
        # write_json(unit_id, cur_unitData)
        # # #
        # cur_unitData = get_zscore_toTrials(memory_path,
        #                                    key_path_info,
        #                                    unit_name=unit_id,
        #                                    output_path=OUTPUT_PATH,
        #                                    csv_pre_name=current_process().name + "_tempfile_" + CSV_PRENAME,
        #                                    first_cell_flag=first_entry_flag,
        #                                    breakpoint_offset=breakpoint_offset,
        #                                    bin_size_for_zscore=0.1,
        #                                    cur_unitData=cur_unitData,
        #                                    baseline_window_for_zscore=(-2., -1.),
        #                                    response_window_for_zscore=(-2., 4.)
        #                                    )

        # Optional if firing rates were already computed
        # cur_unitData = get_trial_info_only(key_path_info,
        #                                    cur_unitData)

        write_json(unit_id, cur_unitData)

        '''
        THE FOLLOWING BLOCK OF FUNCTIONS IS ONLY RELEVANT FOR ACTIVE SESSIONS
        '''
        if 'Passive' not in key_f:
            cur_unitData = calculate_auROC_spoutOffHit(cur_unitData=cur_unitData,
                                                       session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
                                                       pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
                                                       pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
                                                       pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
                                                       post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
                                                       )
            write_json(unit_id, cur_unitData)
            # #
            # cur_unitData = calculate_auROC_hit(cur_unitData=cur_unitData,
            #                                    session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
            #                                    pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                    pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
            #                                    pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                    post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
            #                                    )
            # write_json(unit_id, cur_unitData)
            #
            # cur_unitData = calculate_auROC_FA(cur_unitData=cur_unitData,
            #                                   session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
            #                                   pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                   pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
            #                                   pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                   post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
            #                                   )
            # write_json(unit_id, cur_unitData)
            #
            # cur_unitData = calculate_auROC_missByShock(cur_unitData=cur_unitData,
            #                                            session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
            #                                            pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                            pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
            #                                            pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
            #                                            post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
            #                                            )
            # write_json(unit_id, cur_unitData)


        '''
        COMPUTATIONS RELEVANT TO BOTH PASSIVE AND ACTIVE SESSIONS
        '''
        # cur_unitData = calculate_auROC_AMTrial(cur_unitData=cur_unitData,
        #                                        session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
        #                                        pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
        #                                        pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
        #                                        pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
        #                                        post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
        #                                        )
        # write_json(unit_id, cur_unitData)
        #
        # cur_unitData = calculate_auROC_AMdepthByAMdepth(cur_unitData=cur_unitData,
        #                                                 session_name=split(REGEX_SEP, key_path_info)[-1][:-4],
        #                                                 pre_stimulus_baseline_start=PRETRIAL_DURATION_FOR_SPIKETIMES,
        #                                                 pre_stimulus_baseline_end=PRETRIAL_DURATION_FOR_SPIKETIMES - 1,
        #                                                 pre_stimulus_raster=PRETRIAL_DURATION_FOR_SPIKETIMES,
        #                                                 post_stimulus_raster=POSTTRIAL_DURATION_FOR_SPIKETIMES
        #                                                 )
        # write_json(unit_id, cur_unitData)


def write_json(unit_id, cur_unitData):
    # # Save as json file
    json_filename = unit_id + '_unitData.json'
    with open(OUTPUT_PATH + sep + 'JSON files' + sep + json_filename, 'w') as cur_json:
        cur_json.write(json.dumps(cur_unitData, cls=NumpyEncoder, indent=4))


def compile_fr_result_csv(overwrite_previous):
    master_sheet_name = CSV_PRENAME + '_AMsound_firing_rate.csv'

    process_files = glob(OUTPUT_PATH + sep + '*_tempfile_' + master_sheet_name)

    # Read first process csv just to get the header
    df_header = pd.read_csv(process_files[0], nrows=0)

    # Now read all process csv to compile
    df_merged = (read_csv(f, sep=',', header=None, skiprows=1) for f in process_files)
    df_merged = pd.concat(df_merged, ignore_index=True)

    if overwrite_previous:
        df_header.to_csv(OUTPUT_PATH + sep + master_sheet_name, mode='w', header=True, index=False)

    df_merged.to_csv(OUTPUT_PATH + sep + master_sheet_name, mode='a', header=False, index=False)

    [remove(f) for f in process_files]


def compile_spout_result_csv(overwrite_previous):
    master_sheet_name = CSV_PRENAME + '_allSpoutOnsetOffset_firing_rate.csv'

    process_files = glob(OUTPUT_PATH + sep + '*_' + master_sheet_name)

    # Read first process csv just to get the header
    df_header = pd.read_csv(process_files[0], nrows=0)

    # Now read all process csv to compile
    df_merged = (read_csv(f, sep=',', header=None, skiprows=1) for f in process_files)
    df_merged = pd.concat(df_merged, ignore_index=True)

    if overwrite_previous:
        df_header.to_csv(OUTPUT_PATH + sep + master_sheet_name, mode='w', header=True, index=False)

    df_merged.to_csv(OUTPUT_PATH + sep + master_sheet_name, mode='a', header=False, index=False)

    [remove(f) for f in process_files]


"""
Set global paths and variables
"""
warnings.filterwarnings("ignore")

SPIKES_PATH = '.' + sep + sep.join(['Data', 'Spike times'])
KEYS_PATH = '.' + sep + sep.join(['Data', 'Key files'])
OUTPUT_PATH = '.' + sep + sep.join(['Data', 'Output'])
BREAKPOINT_PATH = '.' + sep + sep.join(['Data', 'Breakpoints'])

BASELINE_DURATION_FOR_FR = 1  # in seconds; for firing rate calculation to non-AM trials
STIM_DURATION_FOR_FR = {'Hit': 1, 'FA': 1, 'Miss': 0.95}   # in seconds; for firing rate calculation to AM trials; ignore shock artifact starting early
AFTERTRIAL_FR_START = {'Hit': 1, 'FA': 1, 'Miss': 1.3}  # Shock artifact is ~0.3s long
AFTERTRIAL_FR_END = {'Hit': 2, 'FA': 2, 'Miss': 2.25}  # 0.95 duration during misses to keep window consistent
PRETRIAL_DURATION_FOR_SPIKETIMES = 2  # in seconds; for grabbing spiketimes around AM trials
POSTTRIAL_DURATION_FOR_SPIKETIMES = 5  # in seconds; for grabbing spiketimes around AM trials
NUMBER_OF_CORES = 4 * cpu_count() // 5
# NUMBER_OF_CORES = 1
# Only run these cells/su or None to run all

CELLS_TO_RUN = None # You can also specify part of the cell file name, like the recording session name
SUBJECTS_TO_RUN = None

DEBUG_RUN = False

overwrite_previous_csv = True

RECORDING_TYPE_DICT = {
    'SUBJ-ID-197': 'synapse',
    'SUBJ-ID-151': 'synapse',
    'SUBJ-ID-154': 'synapse',
    'SUBJ-ID-231': 'intan',
    # 'SUBJ-ID-232': 'intan',
    # 'SUBJ-ID-270': 'intan',
    'SUBJ-ID-389': 'intan',
    'SUBJ-ID-390': 'intan'
}

CSV_PRENAME = 'OFCPL'  # for firing rate calculations
makedirs(OUTPUT_PATH + sep + 'JSON files', exist_ok=True)

if __name__ == '__main__':
    # Load existing JSONs; will be empty if this is the first time running
    all_json = glob(OUTPUT_PATH + sep + 'JSON files' + sep + '*json')

    # Clear temp files if they exist
    process_tempfiles = glob(OUTPUT_PATH + sep + '*_tempfile_*.csv')
    [remove(f) for f in process_tempfiles]

    # Generate a list of inputs to be passed to each worker
    input_lists = list()
    memory_paths = glob(SPIKES_PATH + sep + '*cluster*.txt')
    for dummy_idx, memory_path in enumerate(memory_paths):

        if CELLS_TO_RUN is not None:
            if any([chosen for chosen in CELLS_TO_RUN if chosen in memory_path]):
                pass
            else:
                continue

        if SUBJECTS_TO_RUN is not None:
            if any([chosen for chosen in SUBJECTS_TO_RUN if chosen in memory_path]):
                pass
            else:
                continue

        if DEBUG_RUN:
            run_pipeline((memory_path, all_json))
        else:
            input_lists.append((memory_path, all_json))

    if not DEBUG_RUN:
        pool = Pool(NUMBER_OF_CORES)

        # # Feed each worker with all memory paths from one unit
        pool_map_result = pool.map(run_pipeline, input_lists)

        pool.close()

        pool.join()

        # compile_fr_result_csv(overwrite_previous_csv)
        # compile_spout_result_csv(overwrite_previous_csv)
