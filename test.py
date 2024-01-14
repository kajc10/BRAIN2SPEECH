import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import IEEGDataset
from network import SimpleIEEGModel
from pytorch_lightning import Trainer
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the IEEG Model.')
    parser.add_argument('--checkpoint', type=str, required=False, default='checkpoints/ieeg_model-epoch=38-val_loss=2.16.ckpt', help='Path to the model checkpoint file.')
    args = parser.parse_args()

    # # Load Test Data
    # data_path = "data/preprocessed"
    # participants = ['sub-%02d' % i for i in range(1, 11)]
    # test_dataset = IEEGDataset(feat_path=data_path, participants=participants, preprocess_again=False)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_dataset = torch.load('data/test_dataset.pth')
    test_loader = DataLoader(test_dataset, batch_size=2998, shuffle=False, num_workers=4)
    checkpoint_path =  args.checkpoint #"checkpoints/ieeg_model-epoch=23-val_loss=2.29.ckpt" 


    # Load Trained Model
    # Load the model with the required arguments
    model = SimpleIEEGModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        in_ch=50,
        hidden_dim=128,
        spectrogram_dim=23
    )

    # Logger
    logger = TensorBoardLogger('tb_logs', name='ieeg_model_re')

    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(devices="auto", accelerator="gpu" if torch.cuda.is_available() else "cpu",max_epochs=2,logger=logger)

    # Test the model
    trainer.test(model, dataloaders=test_loader)
