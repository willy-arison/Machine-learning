import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Accuracy
from torch import nn
import torch


class Classifier(pl.LightningModule):
    def __init__(self, model, num_classes=100):
        super().__init__()
        self.model = model
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)  # Forward pass
        self.test_accuracy(preds, y)  # Update accuracy metric
        return preds

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True)


def create_trainer(model_name, max_epochs=5):
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    return pl.Trainer(
        accelerator="gpu",
        strategy="auto",
        precision="16-mixed",
        devices=-1,
        max_epochs=max_epochs,
        logger=pl.loggers.TensorBoardLogger('logs/', name=model_name),
        # check_val_every_n_epoch=4,
        callbacks=[
            # EarlyStopping(monitor="val_acc", mode="max", patience=4),
            # Add this callback to save the best model
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",        # Metric to monitor
                mode="max",               # Save when max accuracy
                save_top_k=1,             # Save only the best model
                filename="{epoch}-{val_acc:.4f}",  # Include accuracy in filename
                save_last=False,          # Don't save final epoch if not best
                verbose=True              # Print when new best model is saved
            )
        ],
    )

class CNN(nn.Module):
    def __init__(self, activation_fn=nn.ReLU(), in_channels=1, num_classes=100):
        super(CNN, self).__init__()

        # Convolutional network
        self.sequential1 = nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),

            # Convolution 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),
            )

        # Fully connected network
        self.sequential2 = nn.Sequential(
            nn.Linear(7*7*64, 128),
            nn.BatchNorm1d(128),
            activation_fn,
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.sequential1(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        return self.sequential2(x)


class MLP(nn.Module):
    def __init__(self, activation_fn=nn.ReLU(), in_channels=1, num_classes=10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.BatchNorm1d(256),
            activation_fn,

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            activation_fn,

            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.sequential(x)

if __name__=='__main__':
    x = torch.randn(16, 28, 28)
    print(MLP()(x))
