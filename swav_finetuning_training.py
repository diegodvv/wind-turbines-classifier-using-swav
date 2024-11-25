import multiprocessing

import random
import numpy as np
import torch
import pytorch_lightning as pl


import os


from sklearn.model_selection import train_test_split

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

import seaborn as sns

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import torch
import torchvision
import pytorch_lightning as pl

import torchvision
from torch import nn

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.transforms.swav_transform import SwaVTransform

class SwaVQueue(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Use the pretrained SwAV ResNet model as the backbone
        resnet = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
        
        # Freeze the backbone, except the last layers, up to (including) the last convolutional layer
        ################################################################################################
        self.backbone.eval()
        
        
        backbone_layers = list(self.backbone.children())
        
        backbone_layers_to_train = [backbone_layers[-1]]

        last_layer_of_sequential = list(backbone_layers[-2].children())[-1]
        backbone_layers_to_train += [last_layer_of_sequential.relu, last_layer_of_sequential.bn3, last_layer_of_sequential.conv3]
        
        for layer in backbone_layers_to_train:
            layer.train()
        ################################################################################################
        self.projection_head = SwaVProjectionHead(2048, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512, 1)
        self.start_queue_at_epoch = 2
        self.queues = nn.ModuleList(
            [MemoryBankModule(size=(3840, 128)) for _ in range(2)]
        )
        self.criterion = SwaVLoss()
        
        # Initialize losses list for tracking loss over epochs
        self.losses = []

    def training_step(self, batch, batch_idx):
        views = batch[0]
        high_resolution, low_resolution = views[:2], views[2:]
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, self.current_epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features)
        loss = self.criterion(
            high_resolution_prototypes, low_resolution_prototypes, queue_prototypes
        )
        
        # Store the loss value for plotting later
        self.log("train_loss", loss, prog_bar=True)
        self.losses.append(loss)

        return loss
    
    def on_epoch_end(self):
        # Calculate and print average loss at the end of each epoch
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0.0
        print(f"Epoch {self.current_epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Clear losses for the next epoch
        self.losses.clear()
        self.log("avg_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optim

    def _subforward(self, input):
        features = self.backbone(input).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if (
            self.start_queue_at_epoch > 0
            and self.current_epoch < self.start_queue_at_epoch
        ):
            return None

        # Assign prototypes
        queue_prototypes = [
            self.prototypes(x, self.current_epoch) for x in queue_features
        ]
        return queue_prototypes
    

# Reads the image from the filesystem when needed, to avoid overflowing RAM
class AirbusDataset(Dataset):
  def __init__(self, background_filenames, target_filenames, transform=None, target_transform=None):
    all_filenames_with_label = [(f,0) for f in background_filenames] + \
                               [(f,1) for f in target_filenames]
    self.filenames = [file_name_label[0] for file_name_label in all_filenames_with_label]
    self.labels = [file_name_label[1] for file_name_label in all_filenames_with_label]

    self.images = self.filenames
    # self.images = [Image.open(filename) for filename in filenames]
    # self.images = [np.asarray(pil_image) for pil_image in self.images]

    self.transform = transform
    self.target_transform = target_transform

    #   self.images = [transform(image) for image in self.images]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    '''
    Return a tuple with the image and the respective target at position idx.
    '''
    image = Image.open(self.images[idx])
    label = self.labels[idx]

    if self.transform != None:
      image = self.transform(image)
      
    if self.target_transform != None:
      label = self.target_transform(label)

    return image, label

def target_transform(t):
    return 0

def main():
    SEED = 172
    # Set seed for Python's random module
    random.seed(SEED)

    # Set seed for NumPy
    np.random.seed(SEED)

    # Set seed for PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.

    # Set seed for PyTorch Lightning
    pl.seed_everything(SEED)

    dataset_folder = 'C:\\Users\\diego\\Documents\\TEMP_MC959\\Airbus Wind Turbines Patches'

    base_train_background_filenames = [os.path.join(dataset_folder, 'train', 'background', file) for file in os.listdir(f'{dataset_folder}/train/background')]
    base_train_target_filenames = [os.path.join(dataset_folder, 'train', 'target', file) for file in os.listdir(f'{dataset_folder}/train/target')]
    base_val_background_filenames = [os.path.join(dataset_folder, 'val', 'background', file) for file in os.listdir(f'{dataset_folder}/val/background')]
    base_val_target_filenames = [os.path.join(dataset_folder, 'val', 'target', file) for file in os.listdir(f'{dataset_folder}/val/target')]

    all_background_filenames = base_train_background_filenames + base_val_background_filenames
    all_target_filenames = base_train_target_filenames + base_val_target_filenames

    # Split the entire dataset into a dataset used for fine-tuning and the rest (used for later stages)
    fine_tuning_background_filenames, rest_background_filenames = train_test_split(all_background_filenames, test_size=0.25, random_state=SEED)
    fine_tuning_target_filenames, rest_target_filenames = train_test_split(all_target_filenames, test_size=0.25, random_state=SEED)

    # Split the rest dataset into 70% train, 15% test, 15% validation
    train_background_filenames, rest_background_filenames = train_test_split(rest_background_filenames, test_size=0.3, random_state=SEED)
    valid_background_filenames, test_background_filenames = train_test_split(rest_background_filenames, test_size=0.5, random_state=SEED)

    train_target_filenames, rest_target_filenames = train_test_split(rest_target_filenames, test_size=0.3, random_state=SEED)
    valid_target_filenames, test_target_filenames = train_test_split(rest_target_filenames, test_size=0.5, random_state=SEED)

    transform = SwaVTransform()

    dataset = AirbusDataset(
        fine_tuning_background_filenames,
        fine_tuning_target_filenames,
        transform=transform,
        target_transform=target_transform,
    )
    
    if not torch.cuda.is_available():
      print("GPU not available!")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    SwaVmodel = SwaVQueue()


    torch.set_float32_matmul_precision('medium')

    # Visualize the results with `tensorboard --logdir=./finetuning/tensorboard_logs/`
    # checkpoint_path = 'finetuning/tensorboard_logs/swav_finetuning/version_14/checkpoints/epoch=2-step=26451.ckpt'

    print("Training the model...")

    logger = TensorBoardLogger('finetuning/tensorboard_logs', name=f"swav_finetuning")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=20,
        devices=1,
        accelerator=accelerator,
        default_root_dir='./finetuning',
        callbacks=[
            ModelCheckpoint(
                # Save all checkpoints
                save_top_k=99999,
                save_last=True,
                monitor="train_loss",
                mode="min",
                # Save checkpoints every quarter epoch
                every_n_train_steps=(len(dataset) // 36) // 8,
                filename="{epoch:02d}-{step:02d}-{train_loss}",
            ),
            EarlyStopping(monitor="train_loss", mode="min", patience=7),
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=3,
    )

    trainer.fit(
        model=SwaVmodel,
        train_dataloaders=dataloader,
        ckpt_path="finetuning/tensorboard_logs/swav_finetuning/version_30/checkpoints/last.ckpt"
    )


if __name__ == '__main__':
    main()
