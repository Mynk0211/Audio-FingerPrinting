
from  typing import List,Tuple,Dict
from itertools import groupby
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from operator import itemgetter
import hashlib





PEAK_SORT = True
num_neighbours = 5
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20
PEAK_NEIGHBORHOOD_SIZE = 10
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_FAN_VALUE = 5
DEFAULT_AMP_MIN = 10
CONNECTIVITY_MASK = 2
SONG_ID = "song_id"
SONG_NAME = 'song_name'
RESULTS = 'results'

HASHES_MATCHED = 'hashes_matched_in_input'

# Hashes fingerprinted in the db.
FINGERPRINTED_HASHES = 'fingerprinted_hashes_in_db'
# Percentage regarding hashes matched vs hashes fingerprinted in the db.
FINGERPRINTED_CONFIDENCE = 'fingerprinted_confidence'

# Hashes generated from the input.
INPUT_HASHES = 'input_total_hashes'
# Percentage regarding hashes matched vs hashes from the input.
INPUT_CONFIDENCE = 'input_confidence'

OFFSET = 'offset'
OFFSET_SECS = 'offset_seconds'

def fingerprint(channel_samples: List[int],
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.
    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    # FFT the signal and extract frequency components
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    local_maxima = get_2D_peaks(arr2D, amp_min=amp_min)
    
    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)



# completed
def get_2D_peaks(arr2D: np.array,  amp_min: int = DEFAULT_AMP_MIN):
    """
    Extract maximum peaks from the spectogram matrix (arr2D).
    :param arr2D: matrix representing the spectogram.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)

 
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D


    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]
    return list(zip(freqs_filter, times_filter))



## completed

def generate_hashes(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE):
 
    # frequencies are in the first position of the tuples
    idx_freq = 0
    # times are in the second position of the tuples
    idx_time = 1

    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))
                    hashes.append((h.hexdigest()[0:FINGERPRINT_REDUCTION], t1))

    return hashes







def getIndexes(dfObj, value, specific_col = None):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin(value)
    # Get list of columns that contains the value
    seriesObj = result.any()
    if not(specific_col):
      columnNames = list(seriesObj[seriesObj == True].index)
    else:
      columnNames = [specific_col]
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append(row)
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos






def return_matches(df_fp, hashes: List[Tuple[str, int]], batch_size: int = 1000):
    """
    Searches the database for pairs of (hash, offset) values.
    :param hashes: A sequence of tuples in the format (hash, offset)
        - hash: Part of a sha1 hash, in hexadecimal format
        - offset: Offset this hash was created from/at.
    :param batch_size: number of query's batches.
    :return: a list of (sid, offset_difference) tuples and a
    dictionary with the amount of hashes matched (not considering
    duplicated hashes) in each song.
        - song id: Song identifier
        - offset_difference: (database_offset - sampled_offset)
    """
    # Create a dictionary of hash => offset pairs for later lookups
    mapper = {}
    for hsh, offset in hashes:
        if hsh in mapper.keys():
            mapper[hsh].append(offset)
        else:
            mapper[hsh] = [offset]

    values = list(mapper.keys())

    # in order to count each hash only once per db offset we use the dic below
    dedup_hashes = {}

    results = []
    
    for index in range(0, len(values), batch_size):
        
        temp = values[index:index + batch_size]
        matches = getIndexes(df_fp, temp)
        
        for indexxie in matches:
            hsh = df_fp.iloc[indexxie]['hash']
            sid = df_fp.iloc[indexxie]['song_id']
            offset = df_fp.iloc[indexxie]['offset(sec)']
            if sid not in dedup_hashes.keys():
                dedup_hashes[sid] = 1
            else:
                dedup_hashes[sid] += 1
            #  we now evaluate all offset for each  hash matched
            for song_sampled_offset in mapper[hsh]:
                results.append((sid, offset - song_sampled_offset))

    return results, dedup_hashes





    
def align_matches(DfObj,songDf, matches: List[Tuple[int, int]], dedup_hashes: Dict[str, int], queried_hashes: int,
                  topn: int = 5):
    """
    Finds hash matches that align in time with other matches and finds
    consensus about which hashes are "true" signal from the audio.
    :param matches: matches from the database
    :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
    (key is the song id).
    :param queried_hashes: amount of hashes sent for matching against the db
    :param topn: number of results being returned back.
    :return: a list of dictionaries (based on topn) with match information.
    """
    # count offset occurrences per song and keep only the maximum ones.
    sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
    counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
    songs_matches = sorted(
        [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
        key=lambda count: count[2], reverse=True
    )

    songs_result = []
    for song_id, offset, _ in songs_matches[0:topn]:  # consider topn elements in the result
        song = getIndexes(DfObj, [song_id], 'song_id')
        song_name = songDf.iloc[song_id]['song_name']
        song_hashes = len(song)
        nseconds = round(float(offset) / DEFAULT_FS * DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5)
        hashes_matched = dedup_hashes[song_id]

        song = {
            SONG_ID: song_id,
            SONG_NAME: song_name.encode("utf8"),
            INPUT_HASHES: queried_hashes,
            FINGERPRINTED_HASHES: song_hashes,
            HASHES_MATCHED: hashes_matched,
            # Percentage regarding hashes matched vs hashes from the input.
            INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 5),
            # Percentage regarding hashes matched vs hashes fingerprinted in the db.
            FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 5),
            OFFSET: offset,
            OFFSET_SECS: nseconds,
            # FIELD_FILE_SHA1: song.get(FIELD_FILE_SHA1, None).encode("utf8")
        }

        songs_result.append(song)

    return songs_result