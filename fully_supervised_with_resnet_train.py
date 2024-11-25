import multiprocessing

import random
import numpy as np
import torch
import pytorch_lightning as pl


import os


from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

import seaborn as sns
from sklearn.metrics import confusion_matrix

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import seaborn as sns
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

class SupervisedResNetCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Instantiate the ResNet backbone
        resnet = torchvision.models.resnet18(pretrained=False)
        self.network = nn.Sequential(
            nn.Sequential(*list(resnet.children())[:-1]),
            nn.Flatten(start_dim=1, end_dim=-1),
            # nn.Linear(in_features=1000, out_features=864, bias=True),
            # nn.ReLU(),
            nn.Linear(in_features=512, out_features=120, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=60, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=10, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2, bias=True),
        )

        self._initialize_weights()

        # Initialize losses and accuracies list for tracking loss over epochs
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        # Store predictions and labels for confusion matrix
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

    def _initialize_weights(self):
      '''
      Initialize the network weights using the Xavier initialization.
      '''
      for x in self.modules():
        if isinstance(x, nn.Linear):
          torch.nn.init.xavier_uniform_(x.weight.data)
          if (x.bias is not None):
            x.bias.data.zero_()

    def forward(self, x):
        logits = self.network(x).flatten(start_dim=1)
        return logits

    def criterion(self, preds, targets): # , device):
      '''
      Function that calculates the loss and accuracy of a batch predicted by the model.
      '''
      ce            = nn.CrossEntropyLoss() # .to(device) # You don't need to change the loss function (but you can if it makes sense on your analysis)
      loss          = ce(preds, targets.long())
      pred_labels   = torch.max(preds.data, 1)[1] # same as argmax
      acc           = torch.sum(pred_labels == targets.data)
      n             = pred_labels.size(0)
      acc           = acc/n

      return loss, acc

    def training_step(self, batch, batch_idx):
        ims, targets = batch
        self.forward(ims)
        preds = self.network(ims)
        loss, acc = self.criterion(preds, targets) #, accelerator)

        # Store the loss value for plotting later
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ims, targets = batch
        self.forward(ims)
        preds = self.network(ims)
        loss, acc = self.criterion(preds, targets) #, accelerator)

        self.val_preds.append(preds.argmax(dim=1))
        self.val_labels.append(targets)
        
        self.val_losses.append(loss)
        self.val_accuracies.append(acc)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        ims, targets = batch
        self.forward(ims)
        preds = self.network(ims)
        loss, acc = self.criterion(preds, targets) #, accelerator)

        self.test_preds.append(preds.argmax(dim=1))
        self.test_labels.append(targets)
        
        self.test_losses.append(loss)
        self.test_accuracies.append(acc)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)

    def on_epoch_end(self):
        # Calculate and print average loss at the end of each epoch
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0.0
        avg_accuracy = sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0.0
        print(f"Epoch {self.current_epoch + 1}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

        # Clear losses, acuracies for the next epoch
        self.losses.clear()
        self.accuracies.clear()
        
    def on_validation_epoch_end(self):
        val_preds = torch.cat(self.val_preds)
        val_labels = torch.cat(self.val_labels)

        # Compute confusion matrix
        cm = confusion_matrix(val_labels.cpu().numpy(), val_preds.cpu().numpy())
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        # Clear stored predictions and labels for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()
    
    def on_test_epoch_end(self):
        test_preds = torch.cat(self.test_preds)
        test_labels = torch.cat(self.test_labels)

        # Compute confusion matrix
        cm = confusion_matrix(test_labels.cpu().numpy(), test_preds.cpu().numpy())
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)

        # Clear stored predictions and labels for the next epoch
        self.test_preds.clear()
        self.test_labels.clear()

    def plot_confusion_matrix(self, cm, test=False):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")

        # Log confusion matrix to TensorBoard
        self.logger.experiment.add_figure(f"{"Test" if test else "Validation"} Confusion Matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim

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


    preprocess_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AirbusDataset(
        train_background_filenames,
        train_target_filenames,
        transform=preprocess_transform
      )

    valid_dataset = AirbusDataset(
        valid_background_filenames,
        valid_target_filenames,
        transform=preprocess_transform
      )

    test_dataset = AirbusDataset(
        test_background_filenames,
        test_target_filenames,
        transform=preprocess_transform
      )
    

    if not torch.cuda.is_available():
      print("GPU not available!")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    supervised_resnet_model = SupervisedResNetCNN()

    # Test
    supervised_resnet_model.eval()
    supervised_resnet_model.forward(train_dataset[0][0].unsqueeze(0))

    torch.set_float32_matmul_precision('medium')

    # checkpoints_folder = '/content/gdrive/MyDrive/MC959 - Projeto/supervised_checkpoints'
    checkpoints_folder = './supervised_checkpoints'
    tensorboard_logs = checkpoints_folder + '/tensorboard_logs'

    MODEL_NAME = "fully_supervised_with_resnet_on_script"
    logger = TensorBoardLogger(tensorboard_logs, name=MODEL_NAME)

    os.makedirs(checkpoints_folder, exist_ok=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=100,
        devices=1,
        accelerator=accelerator,
        default_root_dir=checkpoints_folder,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
      )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        # drop_last=True,
        persistent_workers=True,
        num_workers=10,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        persistent_workers=True,
        num_workers=4,
    )

    supervised_resnet_model.train()
    trainer.fit(
        model=supervised_resnet_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        # ckpt_path="/content/gdrive/MyDrive/MC959 - Projeto/supervised_checkpoints/tensorboard_logs/fully_supervised_with_resnet_on_script/version_11/checkpoints/epoch=7-step=4632.ckpt"
        ckpt_path="./supervised_checkpoints/tensorboard_logs/fully_supervised_with_resnet_on_script/version_11/checkpoints/epoch=7-step=4632.ckpt"
      )
    
    # print(train_dataset[0][0].shape)
    # print(next(iter(train_dataloader)))


if __name__ == '__main__':
    main()
