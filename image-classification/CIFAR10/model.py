import torch.nn.functional as F
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Accuracy
from torch import nn
import torch


class Classifier(pl.LightningModule):
    def __init__(self, model, num_classes=10):
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
        # check_val_every_n_epoch=3,
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
    def __init__(self, activation_fn=nn.ReLU(), img_dim=32, num_classes=10):
        super(CNN, self).__init__()

        # Convolutional network
        self.sequential1 = nn.Sequential(
            # Convolution 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_fn,

            # Convolution 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),

            # Convolution 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),

            # Convolution 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),

            # Convolution 5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            activation_fn,
            nn.MaxPool2d(kernel_size=2),
            )

        # Fully connected network
        self.sequential2 = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.sequential1(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        return self.sequential2(x)



#################ResNet#####################
def conv3x3(in_channels, out_channels, stride=1):
    """
    return 3x3 Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Initialize basic ResidualBlock with forward propogation
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation_fn=nn.ReLU()):
        super(ResidualBlock, self).__init__()

        # Activation function
        self.activation = activation_fn

        # Convolutional 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Convolutional 2
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        # Residual connection
        residual = x

        # Conv layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # Conv layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    """
    Initialize  ResNet with forward propogation
    """
    def __init__(self, block, layers, num_classes=10, activation_fn=nn.ReLU()):
        super(ResNet, self).__init__()

        # Activation function
        self.activation = activation_fn

        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, activation_fn=self.activation))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, activation_fn=self.activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
