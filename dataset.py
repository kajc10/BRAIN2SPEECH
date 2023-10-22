import os
import numpy as np

#dict_keys(['feat.npy', 'feat_names.npy', 'audio_path', 'procWords.npy', 'spec.npy'])
def load_data(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # Extract the subject name (e.g., 'sub-01') and file type (e.g., 'feat_names')
            subject_name, file_type = filename.split("_", 1)
            
            # Load the .npy data
            file_data = np.load(os.path.join(directory, filename))
            
            # Add the loaded data to the dictionary
            if subject_name not in data:
                data[subject_name] = {}
            data[subject_name][file_type] = file_data
        
        elif filename.endswith(".wav"):
            # For audio files, you can store the file path
            subject_name = filename.split("_", 1)[0]
            if subject_name not in data:
                data[subject_name] = {}
            data[subject_name]['audio_path'] = os.path.join(directory, filename)

    return data

# Use the function to load data from the 'preprocessed' directory


if __name__ == "__main__":
    data = load_data('data/preprocessed')

    # Example: To access 'sub-01' features
    print(data['sub-01']['feat.npy'])
    









'''
Note to myself
EG Data Format

The EEG data format might differ slightly depending on the acquisition device and preprocessing steps, but typically it consists of the following:

Channels: EEG data is collected from multiple electrodes placed on the scalp. Each electrode's reading is referred to as a channel. A common EEG might have anywhere from a few channels to 256 channels or more.

Sampling Rate: This is the rate at which data is recorded. Common sampling rates in EEG are 250 Hz, 500 Hz, or 1000 Hz, which means the EEG device is recording data 250, 500, or 1000 times per second, respectively.

Time Series Data: For each channel, you'll get a time series of voltage values corresponding to the brain's electrical activity. This time series data, when plotted, will show the various brain waves.

Events/Markers: These are timestamps indicating when a specific event occurred, such as the start of a stimulus or a particular action by the participant. They are used for event-related potential (ERP) studies where the neural response to specific events is analyzed.

Metadata: This includes information about the participant, electrode locations, acquisition parameters, and any preprocessing steps that have been applied.





Fields:
  acquisition: {
    Audio <class 'pynwb.base.TimeSeries'>,
    Stimulus <class 'pynwb.base.TimeSeries'>,
    iEEG <class 'pynwb.base.TimeSeries'>
  }
  file_create_date: [datetime.datetime(2022, 3, 2, 11, 18, 50, 246485, tzinfo=tzoffset(None, 3600))]
  identifier: sub-01
  session_description: Speech production single words
  session_start_time: 2020-01-01 12:00:00+01:00
  timestamps_reference_time: 2020-01-01 12:00:00+01:00


    acquisition: {
{'Audio': Audio pynwb.base.TimeSeries at 0x2697621036464
Fields:
  comments: no comments
  continuity: continuous
  conversion: 1.0
  data: <HDF5 dataset "data": shape (14414532,), type "<f8">
  description: recorded audio aligned to sEEG data
  interval: 1
  offset: 0.0
  resolution: -1.0
  timestamps: <HDF5 dataset "timestamps": shape (14414532,), type "<f8">
  timestamps_unit: seconds
, 'Stimulus': Stimulus pynwb.base.TimeSeries at 0x2697621036992
Fields:
  comments: no comments
  continuity: continuous
  conversion: 1.0
  data: <StrDataset for HDF5 dataset "data": shape (307511,), type "|O">
  description: word presented on screen during each sEEG sample
  interval: 1
  offset: 0.0
  resolution: -1.0
  timestamps: <HDF5 dataset "timestamps": shape (307511,), type "<f8">
  timestamps_unit: seconds
, 'iEEG': iEEG pynwb.base.TimeSeries at 0x2697621037664
Fields:
  comments: no comments
  continuity: continuous
  conversion: 1.0
  data: <HDF5 dataset "data": shape (307511, 127), type "<f8">
  description: sEEG data
  interval: 1
  offset: 0.0
  resolution: -1.0
  timestamps: <HDF5 dataset "timestamps": shape (307511,), type "<f8">
  timestamps_unit: seconds
  unit: µV
}






'''