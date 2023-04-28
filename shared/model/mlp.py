import torch.nn as nn
import torch.nn.functional as F

from shared.model.common import use_cuda


class MLP(nn.Module):
    def __init__(self, input_shape: int = 2329, output_shape: int = 22):
        super().__init__()
        self.linear_1 = nn.Linear(input_shape, 1000)
        self.linear_2 = nn.Linear(1000, 500)
        self.linear_3 = nn.Linear(500, output_shape)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)

        return x, F.softmax(x, dim=0)


def get_mlp_model(input_shape: int = 2329, output_shape: int = 22):
    mlp_model = MLP(input_shape, output_shape)

    if use_cuda:
        mlp_model.cuda()

    return mlp_model
