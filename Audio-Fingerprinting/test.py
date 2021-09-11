from input_processor import*
from fingerprint import*
import pandas as pd





def display_result(result, seconds, no_of_songs_to_display = 5 , file_path='./test'):

    files = find_files(file_path)
    for i in range(no_of_songs_to_display):
        name = result[i]['song_name']
        start = result[i]['offset_seconds']
        end = start + seconds
        print(f'song name : {name}, matched from {start} secs to {end} secs')        
    return




def test_file(input_signal,df_fp, df_st, seconds = 5 , file_path='./test'):
    # using a random chunk of song for testing even whole song can be tested
    inddex = np.random.randint(0, len(input_signal) - seconds * 44100)
    test_input = input_signal[inddex:inddex + seconds * 44100]


    # finding hashes
    test_hashes = fingerprint(test_input)
    test_hashes = [test_hashes[i][0:2] for i in range(len(test_hashes))]
    hashes = set()
    hashes |= set(test_hashes)
    test_hashes = hashes

    # finding the matches and
    match_results, dedup_hashes = return_matches(df_fp, test_hashes)
    # print(dedup_hashes)
    result = align_matches(df_fp, df_st, match_results, dedup_hashes, len(test_hashes))
   
    return display_result(result, seconds, 1 , file_path)







def main():
    print(f'Welcome to Audio detector!')
    print(f'Now loading the necessary files')
    df_fp = pd.read_csv('generated/hashes.csv')
    print(f'Retrieving the song list!')
    df_st = pd.read_csv('generated/songs.csv')
    print('Done!')

    file_path = input("Enter the address to file directory : ")

    
    seconds = 30

    input_signal,_= input_single_file(file_path)

    test_file(input_signal, df_fp, df_st, seconds ,file_path)

    return 0

if __name__ == "__main__":
    main()