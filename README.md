# BRAIN2SPEECH

:warning: Repository under development - current stage: Milestone2 :warning: 
```
Course: BMEVITMMA19 2023/24/1 
Project Topic: BRAIN2SPEECH 
Team Name: Me, Myself and (A)I
Authors: Katica Bozsó (ZE5BJ7)
```

 **BRAIN2SPEECH** aims to turn brain activity into synthesized speech by leveraging deep learning methods on iEEG (Intracranial Electroencephalography) data, collected via scalp electrodes. This project utilizies the **Dataset of Speech Production in intracranial Electroencephalography** `(SingleWordProductionDutch)`, which contains data of 10 participants reading out individual words in Dutch while their intracranial EEG measured from a total of 1103 electrodes. See https://osf.io/nrgx6/ for documentation and https://osf.io/download/g6q5m/ for data source.

Related works: \
https://github.com/neuralinterfacinglab/SingleWordProductionDutch \
https://www.nature.com/articles/s41597-022-01542-9


## Run
### Option1 - docker image
A docker image can be pulled and directly used by commands: `docker pull kajc10/brain2speech`and `docker run -it --rm brain2speech`
The dataset needs to be downloaded

### Option2 - conda env


## Milestone 2
- to preprocess data: `python SingleWordProductionDutch/extract_features.py `
- to test dataset: `python dataset.py`
- to train model: `python train.py`
- to test model: `python test.py`

next steps:
- longer trainings
- multi-task model
- 2audio script
 

## Milestone 1
- data was downloaded from https://osf.io/nrgx6. (also can be acquired via  [direct link](https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f) )
- [SingleWordProductionDutch](https://github.com/neuralinterfacinglab/SingleWordProductionDutch) codebase was used for preprocessing. If new participant data is added, the `SingleWordProductionDutch/extract_features.py` should be used for feature extraction.
- `dataset.py` contains Dataset, during init it prepares data for a dataloader or loads prepared data from file
- `network.py` contians initial dummy network, no serious design choices yet
- initial Docker file provided
- future design plans:
    - `train.py` - retrain model
    - `test.py` - with the help of the trianed model convert iEEG data to audio
    -  extend data with word embeddings, train a multi-task model




