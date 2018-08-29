from tqdm import tqdm
from scipy.io.wavfile import read

import glob
import numpy as np
import os
import random
import tensorflow as tf


def transform_data(directory, samRate, fileCount):
    """ Transforms data necesary for evaluation """

    # Get the data
    dataFiles = sorted(
        glob.glob(
            os.path.join(
                directory,
                '*.wav'
            )
        )
    )
    random.seed(0)
    random.shuffle(dataFiles)
    dataFiles = dataFiles[:fileCount]

    # Create calculation graph
    data = tf.placeholder(tf.float32, [None])
    x = tf.contrib.signal.stft(data, 2048, 128, pad_end=True)
    x_mag = tf.abs(x)
    wav_mel = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=1025,
        samplerate=samRate,
        lower_edge_hertz=40,
        upper_edge_hertz=samRate
    )
    x_mel = tf.matmul(x_mag, wav_mel)
    x_lmel = tf.log(x_mel + 1e-6)
    x_feat = x_lmel

    # Calculate features for the WAV files
    with tf.Session() as sess:
        _x_feats = []
        for dataFile in tqdm(dataFiles):
            _, _x = read(dataFile)

            _x_feats.append(sess.run(x_feat, {x: _x}))

        _x_feats = np.array(_x_feats)

    return _x_feats
