import torch
import torch.nn as nn

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.model = nn.Sequential(
            ## Backbone

            # Convolutional Block 1
            # It accepts a 3x224x224 image tensor
            # It produces 16 feature maps 224x224 (16x224x224)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # 2x2 pooling with stride 2, which sees 16x224x224 tensor
            # Tensor size is halved, output tensor is 16x112x112
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            # Convolutional Block 2
            # It sees 16x112x112 image tensor
            nn.Conv2d(16, 32, 3, padding=1),   # --> 32x112x112
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                # --> 32x56x56
            nn.ReLU(),

            # Convolutional Block 3
            # It sees 32x56x56 image tensor
            nn.Conv2d(32, 64, 3, padding=1),   # --> 64x56x56
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                # --> 64x28x28
            nn.ReLU(),

            # Convolutional Block 4
            # It sees 64x28x28 image tensor
            nn.Conv2d(64, 128, 3, padding=1),  # --> 128x28x28
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),                # --> 128x14x14
            nn.ReLU(),

            # Convolutional Block 5
            # It sees 128x28x28 image tensor
            nn.Conv2d(128, 256, 3, padding=1), # --> 256x14x14
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),                # --> 256x7x7
            nn.ReLU(),

            # Flatten
            nn.Flatten(),                      # --> 1x256x7x7

            # Head
            # Multilayer Perceptron (MLP) accepts flattened layer 1x256x7x7
            nn.Linear(256*7*7, 1024),          # --> 1024
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(1024, 512),              # --> 512
            nn.ReLU(),

            nn.Linear(512, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
