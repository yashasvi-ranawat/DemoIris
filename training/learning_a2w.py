import os
import numpy as np
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class Net(nn.Module):
    """
    Network
    """
    def __init__(self, in_feat, out_feat, hidden_layers: list[int]):
        super(Net, self).__init__()

        hidden_layers.append(out_feat)

        list_seq = [
            nn.Linear(
                in_feat,
                hidden_layers[0]
            ),
            nn.ReLU()
        ]
        for i, num in enumerate(hidden_layers[1:]):
            list_seq.append(
                nn.Linear(
                    hidden_layers[i],
                    num
                )
            )
            list_seq.append(nn.LeakyReLU())

        #list_seq.append(nn.LogSoftmax(-1))
        
        self.layers = nn.Sequential(
            *list_seq
        )

    def forward(self, x):
        return self.layers(x)


class NetModel(pl.LightningModule):

    def __init__(
        self, lr=0.001, warmup=10, max_iters=300, hidden_layers='4,7'
    ):
        super().__init__()
        self.save_hyperparameters()

        hidden_layers = list(map(int, hidden_layers.split(',')))
        self.net = Net(4, 3, hidden_layers)
        self.example_input_array = torch.zeros(1, 4).type('torch.FloatTensor')

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.forward(x)
                
        loss = F.cross_entropy(out, y)
        
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.forward(x)
        
        loss = F.cross_entropy(out, y)

        self.log('val_loss', loss)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.forward(x)
        
        loss = F.cross_entropy(out, y)

        self.log('test_loss', loss)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, iris):
        'Initialization'
        self.xs = iris["data"]
        self.ys = iris["target"]
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]

        # Load data and get label
        x = torch.tensor(self.xs[ID])
        y = torch.tensor(self.ys[ID])
            
        return x.type('torch.FloatTensor'), y.type('torch.LongTensor')


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 test_ratio = 0.1, # ratio out of total samples for testing
                 train_ratio = 1.0, # ratio out of non-test samples for training
                 val_ratio = 0.2, # ratio out of training samples for validation
                 batch_size = 8,
                 num_workers = 4
                ):
        super().__init__()
        
        # Parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        with open("iris.pkl", "rb") as fio:
            iris = pickle.load(fio)

        # Datasets
        if 'partition.pkl' in os.listdir():
            with open('partition.pkl', 'rb') as f:
                partition = pickle.load(f)
        else:
            ids = np.arange(len(iris['data']))

            ids = np.random.permutation(ids).tolist()
            test_ind = int(len(ids) * (1 - test_ratio))
            train_ind = int(test_ind * train_ratio)
            val_ind = int(train_ind * (1 - val_ratio))
            partition = {'train': ids[:val_ind],
                              'val': ids[val_ind:train_ind],
                              'test': ids[test_ind:]}
            with open('partition.pkl', 'wb') as f:
                pickle.dump(partition, f)

        partition['train'] = partition['train'][:int(len(partition['train'])*train_ratio)] 
        partition['val'] = partition['val'][:int(len(partition['val'])*train_ratio)] 
        dataset_sizes = {x: len(partition[x]) for x in ['train', 'val', 'test']}

        print('# Data: train = {}\n'
              '#       validation = {}\n'
              '#       test = {}'.format(dataset_sizes['train'],
                                       dataset_sizes['val'],
                                       dataset_sizes['test'])
             )
        
        self.data_set = {x: Dataset(partition[x], iris)
                         for x in ['train', 'val', 'test']}

    def train_dataloader(self):
        return data.DataLoader(self.data_set['train'],
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.data_set['val'],
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.data_set['test'],
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=False)
    
    
if __name__ == '__main__':
    # Select the device for training (use GPU if you have one)
    CHECKPOINT_PATH = 'ckpts'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data_module = DataModule()
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=120,
        gradient_clip_val=0.1,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_loss'),
            LearningRateMonitor(logging_interval='epoch')
        ],
    )
    # Check whether pretrained model exists. If yes, load it
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "model.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NetModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(421)
        model = NetModel(
            lr=0.001,
            warmup=10,
            hidden_layers="8, 4, 8",
            max_iters=trainer.max_epochs*len(data_module.train_dataloader())
        )
        
    trainer.fit(model, data_module)


