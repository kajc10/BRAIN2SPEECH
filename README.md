# BRAIN2SPEECH

```
Course: BMEVITMMA19 2023/24/1 
Project Topic: BRAIN2SPEECH 
Team Name: Me, Myself and (A)I
Authors: Katica Bozsó (ZE5BJ7)
```
Homework [documentation](docs/brain2speech.pdf).

 **BRAIN2SPEECH** aims to turn brain activity into synthesized speech by leveraging deep learning methods on iEEG (Intracranial Electroencephalography) data, collected via scalp electrodes. This project utilizies the **Dataset of Speech Production in intracranial Electroencephalography** `(SingleWordProductionDutch)`, which contains data of 10 participants reading out individual words in Dutch while their intracranial EEG measured from a total of 1103 electrodes. See https://osf.io/nrgx6/ for documentation and https://osf.io/download/g6q5m/ for data source. 

Related works: \
[1] https://github.com/neuralinterfacinglab/SingleWordProductionDutch - note that minimal modifications were made that solved compatibility issues with newer packages\
[2] https://www.nature.com/articles/s41597-022-01542-9


## Method
 The task is to predict correct spectrograms for neural features. This is formulated as a supervised problem. The dataloader yields feature-spectrogram pairs, a CNN model is applied, and MSE is calculated between the ground-truth spectrogram and the prediction.

Authors of [[1]](https://github.com/neuralinterfacinglab/SingleWordProductionDutch) provided multiple scripts. (Note that minimal modifications were made due to deprecations.)
The feature extraction script, from each participant's iEEG measurement, creates the following files:
- sub-{i}_feat_names.npy - electrode details
- sub-{i}_feat.npy - feautures
- sub-{i}_orig_audio.wav - audio file of words read
- sub-{i}_procWords.npy - wordlist
- sub-{i}_spec.npy - spectogram

This repository's IEEGDataset further processes the features via PCA and windowing. The dataloader handles this processed data.

## Setup
First the environment has to be set up, data needs to be downloaded, and an external preprocess script needs to be run.
### Option1 - conda env
Use the repository directly: 
```
conda create -n brain2speech python=3.11.7
conda activate brain2speech
pip install -r requirements.txt
```
The dataset can be downloaded via [direct link](https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f) or downloader script `download_dataset.sh` (Linux) / `download_dataset.ps1` (Windows; you might need 'Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
').
The dataset shall be placed as `data/SingleWordProductionDutch-iBIDS`.
Run dataset feature extraction provided by the original repository: `python SingleWordProductionDutch/extract_features.py` . It will place data under `data/features`.

### Option2 - build docker image
A docker image can be built directly from the provided Dockerfile. Commands: `docker build -t brain2speech .` and `docker run --gpus all -it -p 6006:6006 -p 22:22 brain2speech`.
To download dataset run: `download_dataset.sh`. It will be placed as `data/SingleWordProductionDutch-iBIDS`.
Run dataset feature extraction provided by the original repository: `python SingleWordProductionDutch/extract_features.py`. It will place data under `data/features`.

### Option3 - pull docker image
A docker image can be pulled and directly used by commands: `docker pull kajc10/brain2speech_dl` and `docker run -it brain2speech_dl`.
To download dataset run: `download_dataset.sh`. It will be placed as `data/SingleWordProductionDutch-iBIDS`.
Run dataset feature extraction provided by the original repository: `python SingleWordProductionDutch/extract_features.py`. It will place data under `data/features`.

## Run
Check CUDA setup:
```
>>> import torch
>>> torch.cuda.is_available()
True
```

- train: At this point data is downloaded, and the authors' feature extraction is carried out. Now, for training, data has to be further modified ( PCA ), and this is handled by the dataset. The preprocessing can be done by running `python train.py --preprocess_again`. This will create `data/features/processed_data.npz` which can be used for training. Note that this has to be done only once, or if new data is added, otherwise you can just run training with `python train.py`
You can check tensorboard logs via `tensorboard --logdir tb_logs --bind_all`
- test: `python dataset.py`. The test script uses a dataset saved by the train script before training - `data/test_dataset.pth`.
Spectograms are predicted from the iEEG data, and besides the loss measurement, speech is synthesized as well.


