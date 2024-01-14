import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset

# builds on https://github.com/neuralinterfacinglab/SingleWordProductionDutch/tree/main
# takes the author's feature extract and further preprocesses it -> creates 'processed_data.npz'
# can recall it anytime (e.g. if more extract was prepared)

# Features extracted:
# - sub-{i}_feat_names.npy - electrode details
# - sub-{i}_feat.npy - feautures
# - sub-{i}_orig_audio.wav - audio file of words read
# - sub-{i}_procWords.npy - wordlist
# - sub-{i}_spec.npy - spectogram

# Produce PCA reducted standardized training data
# return feature-spectrogram pairs

class IEEGDataset(Dataset):
    def __init__(self, feat_path, participants, window_size, preprocess_again=True, transform=None, save_name='processed_data.npz'):
        self.feat_path = feat_path
        self.window_size = window_size
        self.transform = transform
        self.save_name = save_name
        
        self.features = []   #features
        self.spectrograms = [] #spectrogram
        self.participants = participants #['sub-%02d' % i for i in range(1, 11)]
        if preprocess_again:
            print('### Preprocessing data in Dataset Init ###')   
            print('~~participants: ',self.participants)        
            for pt in self.participants:
                feature, spectrogram = self._process_pt_data(pt)
                self.features.append(feature)
                self.spectrograms.append(spectrogram)
            
            self.features = np.concatenate(self.features, axis=0)
            self.spectrograms = np.concatenate(self.spectrograms, axis=0)
            
            # Save the processed data to disk
            np.savez(os.path.join(self.feat_path, self.save_name), 
                     features=self.features, 
                     spectrograms=self.spectrograms)
        else:
            print('### Loading already preprocessed data in Dataset Init ###')
            self._load_data()

    def _process_pt_data(self, pt): #one subject
        features = np.load(os.path.join(self.feat_path, f'{pt}_feat.npy'))
        spectrogram = np.load(os.path.join(self.feat_path, f'{pt}_spec.npy'))

        # Standardize data
        mu = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        features = (features-mu)/std

        # Reduce Dimensions using PCA
        pca = PCA()
        pca.fit(features)
        features = np.dot(features, pca.components_[:50,:].T)

        # Creating windowed data
        total_samples, feat_dims = features.shape
        _, spec_dims = spectrogram.shape
        x_features = np.zeros((total_samples - self.window_size, self.window_size, feat_dims))
        y_spectrograms = np.zeros((total_samples - self.window_size, spec_dims))

        for idx in range(total_samples - self.window_size):
            x_features[idx] = features[idx: idx + self.window_size]
            y_spectrograms[idx] = spectrogram[idx + self.window_size]

        return x_features, y_spectrograms

    def _load_data(self):
        data = np.load(os.path.join(self.feat_path, self.save_name))
        self.features = data['features']
        self.spectrograms = data['spectrograms']
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # TODO also return word embedding for multitask learning
        #sample = self.features[idx], self.spectrograms[idx]
        #if self.transform:
        #    sample = self.transform(sample)  # TODO separate transform for feature and spectrogram
        #return sample
        sample = torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.spectrograms[idx], dtype=torch.float32)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process IEEG dataset.')
    parser.add_argument('--preprocess_again', action='store_true', help='Preprocess the data again (default: False)')
    parser.add_argument('--save_name', default='processed_data.npz',help='save/load to/from this file')
    parser.add_argument('--participants', default='all',help='all or specify')
    args = parser.parse_args()

    data_path = os.path.join('data', 'features')
    if args.participants == 'all':
        participants = ['sub-%02d' % i for i in range(1, 11)] # proc all 10 participants..
    else:
        participant_nums = args.participants.split(',')
        participants = ['sub-%s' % num.zfill(2) for num in participant_nums]  # Ensure two-digit formatting

    dataset = IEEGDataset(feat_path=data_path, participants=participants, window_size=4, preprocess_again=args.preprocess_again,save_name=args.save_name)
    element = dataset.__getitem__(0)
    print('feature: ',element[0])
    print('spectogram',element[1])