# Data

The SingleWordProductionDutch dataset contains `iEEG` measurements for 10 participants in the form of `.nwb` files.


> The data is structured following the BIDS version 1.7.0 specifcation (https://bids-specifcation.readthedocs.io/en/stable/). The root folder contains metadata of the participants (participants.tsv), subject specifc data
folders (i.e., sub-01) and a derivatives folder. The subject specifc folders contain .tsv fles with information
about the implanted electrode coordinates (_electrodes.tsv), recording montage (_channels.tsv) and event markers (_events.tsv). The _ieeg.nwb fle contains three raw data streams as timeseries (iEEG, Audio and Stimulus),
which are located in the acquisition container. Descriptions of recording aspects and of specifc .tsv columns are
provided in correspondingly named .json fles (i.e., participants.json). The derivatives folder contains the pial
surface cortical meshes of the right (_rh_pial.mat) and lef (_lh_pial.mat) hemisphere, the brain anatomy (_
brain.mgz), the Destrieux atlas (_aparc.a2009s+aseg.mgz) and a white matter atlas (_wmparc.mgz) per subject,
derived from the Freesurfer pipeline. The description column in the _channels.tsv fle refers to the anatomical
labels derived from the Destrieux atlas.

Full documentation at: https://www.nature.com/articles/s41597-022-01542-9

For each participant, the authors' preprocess script creates the following files:
- sub-{i}_feat_names.npy - electrode details
- sub-{i}_feat.npy - feautures
- sub-{i}_orig_audio.wav - audio file of words read
- sub-{i}_procWords.npy - wordist
- sub-{i}_spec.npy - spectogram
