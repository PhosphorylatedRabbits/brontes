"""Network used in MNIST examples."""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """A simple network for MNIST."""

    def __init__(self):
        """Initialize the network."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 1)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        """
        Apply the forward pass.

        Args:
            x (torch.tensor): an examples.

        Returns:
            a torch.tensor with the prediction.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
