#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [Essentia](http://essentia.upf.edu) (C++ with Python bindings)
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)
import time
import os
import multiprocessing
import warnings

from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
import utils

import matplotlib.pyplot as plt
import ast
import librosa.display


def make_spec(x,y):
    audio_dir = "/media/ravi/507412DD7412C59E/fma_small/"
    spec_out_dir ="/media/ravi/507412DD7412C59E/spectrogram/"
    aud_path = utils.get_audio_path(audio_dir, x)
    out_path = spec_out_dir + "{}/{}".format(y,x)
    return (song_to_spec(aud_path),out_path)


def loadaudio(fname,window=128):
    sr = 44100
    y=utils.FfmpegLoader(sampling_rate=sr)._load(filename,"/home/ravi/anaconda3/envs/amadeus/bin/")

    y_last = y.shape[0]-y.shape[0]%10
    y=y[:y_last]
    leng = y.shape[0]/10


    breaky = np.split(y,10)
    return breaky


#%matplotlib inline
def save_spec(S,fname):
    plt.ioff()

    librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='off', \
                             fmax=8000,x_axis='off',sr=44100)

    #plt.show()
    fig = plt.gcf()

    fig.set_size_inches(1,1)
    fig.frameon=False

    fig.savefig(fname, bbox_inches='tight',dpi=128,pad_inches=0)
    plt.close(fig)


filename="/media/ravi/507412DD7412C59E/fma_small/000/000005.mp3"
def song_to_spec(infile):
    sr=44100
    break_y=loadaudio(infile)
    S_top = []
    for i in range(len(break_y)):
        S_top.append(librosa.feature.melspectrogram(y=break_y[i], sr=sr, n_mels=128,  fmax=8000))
    return S_top


def load_track(filepath):

    filename = os.path.basename(filepath)
    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]#, ('track', 'genre_top')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                ('album', 'date_created'), ('album', 'date_released'),
                ('artist', 'date_created'), ('artist', 'active_year_begin'),
                ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype('category', categories=SUBSETS, ordered=True)

    COLUMNS = [('track', 'license'), ('artist', 'bio'),('album', 'type'), ('album', 'information')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks

def run(strs):
    import time
    st= time.time()
    q=0

    for x in  train[ (strs<=train)]:
        print(x,end=' ')
        song = make_spec(x,labels[x])
        k=0
        q+=1
        for s in song[0]:
            save_spec(s,song[1]+"_{}".format(k))
            k+=1
        de = time.time()-st

        print(de,(de)/q)
        del song
    


if __name__ =='__main__':
    AUDIO_DIR = "/media/ravi/507412DD7412C59E/fma_small/" #os.environ.get('AUDIO_DIR')

    tracks = load_track('/home/ravi/metafma/fma_metadata/tracks.csv')
    tracks.shape
    features = utils.load('/home/ravi/metafma/fma_metadata/features.csv')
    features.shape

    echonest = utils.load('/home/ravi/metafma/fma_metadata/echonest.csv')
    echonest.shape
    subset = tracks.index[tracks['set', 'subset'] <= 'small']

    assert subset.isin(tracks.index).all()
    assert subset.isin(features.index).all()

    features_all = features.join(echonest, how='inner').sort_index(axis=1)
    print('Not enough Echonest features: {}'.format(features_all.shape))

    tracks = tracks.loc[subset]
    features_all = features.loc[subset]

    tracks.shape, features_all.shape

    train = tracks.index[tracks['set', 'split'] == 'training']
    val = tracks.index[tracks['set', 'split'] == 'validation']
    test = tracks.index[tracks['set', 'split'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    genres = list(MultiLabelBinarizer().fit(tracks['track', 'genre_top']).classes_)
    #genres = list(tracks['track', 'genre_top'].unique())
    print('Top genres ({}): {}'.format(len(genres), genres))
    genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
    print('All genres ({}): {}'.format(len(genres), genres))
    labels = {x:y for x,y in tracks['track','genre_top'].iteritems()}
    ys=6443
    while True:
        ys=run(6443)
