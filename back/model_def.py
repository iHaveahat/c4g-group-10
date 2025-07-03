import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size=28996):  
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(256 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        return self.network(x)
