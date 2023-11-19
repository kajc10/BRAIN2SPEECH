import pytorch_lightning as pl
import torch.nn as nn
import torch

class SimpleIEEGModel(pl.LightningModule):
    def __init__(self, in_ch, hidden_dim, spectrogram_dim): # TODO ethink if wanna use hidden dim for cfg
        super(SimpleIEEGModel, self).__init__()

        # 1D CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128, 128),  
            nn.ReLU(),
            nn.Linear(128, spectrogram_dim)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Adjust input dimensions for 1D convolution
        x = x.permute(0, 2, 1)  # [batch_size, num_features, sequence_length]
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y_spectrogram = batch
        pred_spectrogram = self(x)
        loss = self.loss_fn(pred_spectrogram, y_spectrogram)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_spectrogram = batch
        pred_spectrogram = self(x)
        loss = self.loss_fn(pred_spectrogram, y_spectrogram)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_spectrogram = batch
        pred_spectrogram = self(x)
        loss = self.loss_fn(pred_spectrogram, y_spectrogram)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class MultiTaskIEEGModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, spectrogram_dim, embedding_dim, num_words):
        super(MultiTaskIEEGModel, self).__init__()

        # Shared Encoder
        self.encoder = nn.Sequential(
            # TODO
        )

        # Spectrogram Decoder
        self.spectrogram_decoder = nn.Sequential(
            # TODO      
        )

        # Word Embedding Decoder
        self.embedding_decoder = nn.Sequential(
            # TODO
        )

        # Mean Square Error for spectrogram and cosine similarity for embeddings
        self.loss_spectrogram = nn.MSELoss()
        self.loss_embedding = nn.CosineEmbeddingLoss()

    def forward(self, x):
        backbone = self.encoder(x)
        spectrogram = self.spectrogram_decoder(backbone)
        embedding = self.embedding_decoder(backbone)
        return spectrogram, embedding

    def training_step(self, batch, batch_idx):
        x, y_spectrogram, y_embedding, _ = batch
        pred_spectrogram, pred_embedding = self(x)
        loss1 = self.loss_spectrogram(pred_spectrogram, y_spectrogram)
        
        ones = torch.ones(y_embedding.size(0)).to(self.device) # Assuming you have a tensor of 1s for positive samples
        loss2 = self.loss_embedding(pred_embedding, y_embedding, ones)
        loss = loss1 + loss2
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_spectrogram, y_embedding, _ = batch
        pred_spectrogram, pred_embedding = self(x)
        loss1 = self.loss_spectrogram(pred_spectrogram, y_spectrogram)
        ones = torch.ones(y_embedding.size(0)).to(self.device)
        loss2 = self.loss_embedding(pred_embedding, y_embedding, ones)
        loss = loss1 + loss2
        self.log("val_loss", loss)
        return loss