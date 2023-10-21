import pytorch_lightning as pl
import torch.nn as nn
import torch

class MultiTaskEEGModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, spectrogram_dim, embedding_dim, num_words):
        super(MultiTaskEEGModel, self).__init__()

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

        # Assuming you use Mean Square Error for spectrogram and cosine similarity for embeddings
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