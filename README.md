# BRAIN2SPEECH

:warning: Repository under development - current stage: Milestone1 :warning: 
```
Course: BMEVITMMA19 2023/24/1 
Project Topic: BRAIN2SPEECH 
Team Name: Me, Myself and (A)I
Authors: Katica Bozsó (ZE5BJ7)
```

**BRAIN2SPEECH** aims to turn brain activity into synthesized speech by leveraging deep learning methods on EEG (Electroencephalography) data, collected via scalp electrodes. This deep dive into neural exploration paves the way for future BCI technologies that might bypass the need for invasive electrodes, pushing us towards more seamless human-machine interactions.

**Dataset:** The Dataset of Speech Production in intracranial Electroencephalography `(SingleWordProductionDutch)` contains data of 10 participants reading out individual words in Dutch while their intracranial EEG measured from a total of 1103 electrodes: https://osf.io/nrgx6/.

Related works: \
https://github.com/neuralinterfacinglab/SingleWordProductionDutch \
https://www.nature.com/articles/s41597-022-01542-9

## Milestone 1
- data was downloaded from https://osf.io/nrgx6. (also can be acquired via  [direct link](https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f) )
- [SingleWordProductionDutch](https://github.com/neuralinterfacinglab/SingleWordProductionDutch) codebase was used for preprocessing. If new participant data is added, the `extract_features.py` should be used for feature extraction.
- `dataset.py` contains Dataset, during init it prepares data for a dataloader or loads prepared data from file
- `network.py` contians initial dummy network, no serious design choices yet
- initial Docker file provided
- future design plans:
    - `train.py` - retrain model
    - `test.py` - with the help of the trianed model convert iEEG data to audio
    -  extend data with word embeddings, train a multi-task model




