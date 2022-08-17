import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl

labels = ['setosa', 'versicolor', 'virginica']
inp_labels = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]


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

        # list_seq.append(nn.LogSoftmax(-1))
        
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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = LogWarmupScheduler(
            optimizer,
            warmup=self.hparams.warmup,
            max_iters=self.hparams.max_iters
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.forward(x)
                
        loss = F.nll_loss(out, y)
        
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        out = self.forward(x)
        
        loss = F.nll_loss(out, y)

        self.log('val_loss', loss)


def predict(inp):
    inp = np.array(inp)
    
    if inp.shape != (4,):
        raise ValueError(
            f"bad input shape, expects 4 floats as inputs corresponding to"
            f" [{', '.join(inp_labels)}]"
        )
    net = NetModel.load_from_checkpoint("model.ckpt")
    
    inp = torch.tensor(inp).type("torch.FloatTensor").expand(1, -1)
    output = labels[np.argmax(net.forward(inp).detach().numpy()[0])]

    return output
