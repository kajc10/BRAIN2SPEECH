# BRAIN2SPEECH

```
Course: BMEVITMMA19 2023/24/1 
Project Topic: BRAIN2SPEECH 
Team Name: Me, Myself and (A)I
Authors: Katica Bozsó (ZE5BJ7)
```

BRAIN2SPEECH aims to turn brain activity into synthesized speech by leveraging deep learning methods on EEG (Electroencephalography) data, collected via scalp electrodes. This deep dive into neural exploration paves the way for future BCI technologies that might bypass the need for invasive electrodes, pushing us towards more seamless human-machine interactions.

**Dataset:** Dataset of Speech Production in intracranial Electroencephalography SingleWordProductionDutch) - contains data of 10 participants reading out individual words while we measured intracranial EEG from a total of 1103 electrodes
`https://osf.io/g6q5m`

## Milestone 1

### Run

With Docker:\
Build the container: `docker build -t brain2speech .` \
Run the container: `docker run -it brain2speech` \
Inside the container, execute the scripts/notebooks. \
    - train: `python train.py` 
    - test: `python test.py`

Without Docker: \
Install requirements: `pip install -r requirements.txt` \
train: `python train.py` \
test: `python test.py`