import torch
from torch.utils.data import DataLoader
from dataset import IEEGDataset
from network import SimpleIEEGModel
from pytorch_lightning import Trainer
import os

if __name__ == "__main__":
    # # Load Test Data
    # data_path = "data/preprocessed"
    # participants = ['sub-%02d' % i for i in range(1, 11)]
    # test_dataset = IEEGDataset(feat_path=data_path, participants=participants, preprocess_again=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_dataset = torch.load('data/test_dataset.pth')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    checkpoint_path = "checkpoints/ieeg_model-epoch=23-val_loss=2.29.ckpt" 


    # Load Trained Model
    # Load the model with the required arguments
    model = SimpleIEEGModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        in_ch=50,
        hidden_dim=128,
        spectrogram_dim=23
    )


    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(devices="auto", accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Test the model
    trainer.test(model, dataloaders=test_loader)
