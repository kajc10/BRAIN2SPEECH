# script that reconstructs the audio from the spectrogram
# it is taken from https://github.com/neuralinterfacinglab/SingleWordProductionDutch/blob/main/reconstruction_minimal.py
# reconstruction is not perfect, but the audio is recognizable

import numpy as np
import soundfile as sf
import os
import SingleWordProductionDutch.reconstructWave as rW
import SingleWordProductionDutch.MelFilterBank as mel
import numpy as np
import scipy.io.wavfile as wavfile

def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    nfolds = 10
    hop = int(spectrogram.shape[0]/nfolds)
    rec_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)
    for w in range(0,spectrogram.shape[0],hop):
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio = np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled


if __name__ == "__main__":
    pt = '01'
    output_folder = os.path.join('.','outputs','synthesized_voice')
    os.makedirs(output_folder, exist_ok=True)
    
    spectrogram_path  = os.path.join('data', 'features', f'sub-{pt}_spec.npy')
    spectrogram = np.load(spectrogram_path) 
    audiosr=16000
    winLength=0.05
    frameshift=0.01

    origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
    wavfile.write(os.path.join('synthesized_outputs',f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)