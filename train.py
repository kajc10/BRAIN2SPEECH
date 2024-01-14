import torch
import os
from torch.utils.data import DataLoader, random_split
from dataset import IEEGDataset
from network import SimpleIEEGModel, MultiTaskIEEGModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process IEEG dataset.')
    parser.add_argument('--preprocess_again', action='store_true', help='Preprocess the data again (default: False)')
    parser.add_argument('--save_name', default='processed_data.npz',help='save/load to/from this file',type=str)
    parser.add_argument('--participants', default='all',help='all or specify',type=str)
    parser.add_argument('--batch_size', default=32,help='batch size',type=int)

    args = parser.parse_args()

    data_path = os.path.join('data', 'features')
    if args.participants == 'all':
        participants = ['sub-%02d' % i for i in range(1, 11)] # proc all 10 participants..
    else:
        participant_nums = args.participants.split(',')
        participants = ['sub-%s' % num.zfill(2) for num in participant_nums]  # Ensure two-digit formatting


    dataset = IEEGDataset(feat_path=data_path, participants=participants, preprocess_again=args.preprocess_again, save_name=args.save_name,window_size=4)
    num_train = int(0.7 * len(dataset))
    num_val = int(0.2 * len(dataset))
    num_test = len(dataset) - num_train - num_val

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    torch.save(test_dataset, 'data/test_dataset.pth')


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    batch = next(iter(val_loader))
    print(batch[0].shape, batch[1].shape) #[batch_size, sequence_length, num_features]:[32, 4, 50], [batch_size, num_spectrogram_features]:[32, 23]
    #print(batch[0], batch[1])





    model = SimpleIEEGModel(in_ch=50, hidden_dim=128, spectrogram_dim=23)

    # Logger
    logger = TensorBoardLogger('tb_logs', name='ieeg_model_re')

    # Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='ieeg_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Early stopping
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=False,
    mode='min'
    )

    # Trainer
    trainer = pl.Trainer(logger=logger,
                        callbacks=[checkpoint_callback, early_stop_callback],
                        max_epochs=50,
                        devices="auto",
                        accelerator="gpu" if torch.cuda.is_available() else "cpu")

    trainer.fit(model, train_loader, val_loader)
