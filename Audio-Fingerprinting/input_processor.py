import librosa, librosa.display
import tqdm
from scipy.io import wavfile
import numpy as np
import os
import torch as t



def audio_preprocess(x, aug_blend = True):
    # Extra layer in case we want to experiment with different preprocessing
    # For two channel, blend randomly into mono (standard is .5 left, .5 right)

    x = x.float()
    if x.shape[-1]==2:
        if aug_blend:
            mix=t.rand((x.shape[0],1), device=x.device) #np.random.rand()
        else:
            mix = 0.5
        x=(mix*x[:,:,0]+(1-mix)*x[:,:,1])
    elif x.shape[-1]==1:
        x=x[:,:,0]
    else:
        assert False, f'Expected channels {hps.channels}. Got unknown {x.shape[-1]} channels'

    # x: NT -> NTC
    x = x.unsqueeze(2)
    return x




def input_processor(files_dir, num_files=5):
    print(f'Processing Input!')
    files = librosa.util.find_files(files_dir, ['wav'])
    print(f'Found {len(files)} files!')

    if(len(files) == 0):
        print(f'Please enter a valid directory!')
        return None,None
    files = files[:num_files]
    input_all = []
    input_names = []



    for file in tqdm.tqdm(files):
        # saving the name of the file
        name = os.path.basename(file)
        input_names.append(name[:len(name) - 4])  # subtracting the extension
        # main reading of the wave file
        rate, song_array = wavfile.read(file)  # In future add mp3 to wav converter here maybe (Also look about the tradeoff of conversion)
        # preprocessing
        
        song_array = song_array.reshape(1, song_array.shape[0], song_array.shape[1])
        song_array = t.from_numpy(song_array)
        input_ = audio_preprocess(song_array)
        input_ = input_.reshape(input_.shape[1])
        input_ = input_.numpy()
        input_all.append(input_)

    return input_all , input_names



def input_single_file(file_path):
    input_sig = []
    input_name = []

    files = librosa.util.find_files(file_path, ['wav'])
    if(len(files) > 1):
        return

    file = files[0]
    # saving the name of the file
    name = os.path.basename(file)
    input_name.append(name[:len(name) - 4])  # subtracting the extension
    # main reading of the wave file
    rate, song_array = wavfile.read(file)  # In future add mp3 to wav converter here maybe (Also look about the tradeoff of conversion)
    # preprocessing

    song_array = song_array.reshape(1, song_array.shape[0], song_array.shape[1])
    song_array = t.from_numpy(song_array)
    input_ = audio_preprocess(song_array)
    input_ = input_.reshape(input_.shape[1])
    input_ = input_.numpy()
    input_sig.append(input_)

    return input_sig[0],input_name[0]


def find_files(file_dir = './data'):
    files = librosa.util.find_files(file_dir, ['wav'])
    # print(f'Found {len(files)} files!')

    if (len(files) == 0):
        print(f'Please enter a valid directory!')
        return None, None

    return files