import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
criterion = (
    nn.CrossEntropyLoss()
)  # WOULD NEED TODO: Compute class weights to handle umbalanced data
