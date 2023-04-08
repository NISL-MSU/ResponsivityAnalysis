from abc import ABC
import torch.nn as nn
from torch import reshape, cat, Tensor, matmul, no_grad, optim, rand, index_select


class NN(nn.Module, ABC):
    """Defines NN architecture for yield prediction"""

    def __init__(self, input_shape=500, output_size=1):
        super(NN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.1)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        return self.out(x)


class NNcounterfact(nn.Module, ABC):
    """Defines NN architecture for yield prediction"""

    def __init__(self, input_shape=500, output_size=1):
        super(NNcounterfact, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.1)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

        # Number of secondary outputs depends on the dimensionality of the input minus 1
        self.out2 = nn.Linear(50, input_shape - 1)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        return [self.out(x), self.out2(x)]


class NN2(nn.Module, ABC):
    """Defines NN architecture for the other datasets"""

    def __init__(self, input_shape=500, output_size=1):
        super(NN2, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100), nn.Tanh())
        self.drop2 = nn.Dropout(p=0.1)

        # Number of outputs depends on the method
        self.out = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.drop1(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        return self.out(x)


class Hyper3DNetLiteReg(nn.Module, ABC):
    """Our proposed 3D-2D CNN."""

    def __init__(self, input_shape=(1, 15, 5, 5), output_size=5, output_channels=1):
        super(Hyper3DNetLiteReg, self).__init__()
        # Set stride
        stride = 1
        # If the size of the output patch is less than the input size, don't apply padding at the end
        if output_size < input_shape[2]:
            padding = 0
        else:
            padding = 1

        self.input_shape = input_shape
        self.output_size = output_size
        self.output_channels = output_channels
        nfilters = 32

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(in_channels=input_shape[0], out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 2, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))
        # self.drop0 = nn.Dropout(p=0.5)
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(in_channels=nfilters * 3, out_channels=nfilters, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm3d(nfilters))

        self.drop = nn.Dropout(p=0.1)

        self.sepconv1 = nn.Sequential(
            nn.Conv2d(in_channels=nfilters * 4 * input_shape[1], out_channels=nfilters * 4 * input_shape[1],
                      kernel_size=3, padding=1, groups=nfilters * 4 * input_shape[1]), nn.ReLU(),
            nn.Conv2d(in_channels=nfilters * 4 * input_shape[1], out_channels=512,
                      kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(512))
        self.sepconv2 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512,
                                                kernel_size=3, padding=1, stride=stride, groups=512), nn.ReLU(),
                                      nn.Conv2d(in_channels=512, out_channels=320,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(320))
        self.drop2 = nn.Dropout(p=0.1)
        self.sepconv3 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=320,
                                                kernel_size=3, padding=1, stride=stride, groups=320), nn.ReLU(),
                                      nn.Conv2d(in_channels=320, out_channels=256,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(256))
        self.drop3 = nn.Dropout(p=0.1)
        self.sepconv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256,
                                                kernel_size=3, padding=1, stride=stride, groups=256), nn.ReLU(),
                                      nn.Conv2d(in_channels=256, out_channels=128,
                                                kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm2d(128))
        self.sepconv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=32,
                                                kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))

        # This layer is used in case the outputSize is 1
        if output_size == 1:
            self.out = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1,
                                               kernel_size=3, padding=padding), nn.ReLU())
            self.fc = nn.Linear(9, output_channels)
        else:
            self.out = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=output_channels,
                                               kernel_size=3, padding=padding))

    def forward(self, x):
        # print(device)
        # 3D Feature extractor
        x = self.conv_layer1(x)
        x2 = self.conv_layer2(x)
        x = cat((x, x2), 1)
        x2 = self.conv_layer3(x)
        x = cat((x, x2), 1)
        # x = self.drop0(x)
        x2 = self.conv_layer4(x)
        x = cat((x, x2), 1)
        # Reshape 3D-2D
        x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.drop(x)
        # 2D Spatial encoder
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.drop2(x)
        x = self.sepconv3(x)
        x = self.drop3(x)
        x = self.sepconv4(x)
        x = self.sepconv5(x)
        # Final output
        x = self.out(x)

        # Flatten and apply the last fc layer if the output is just a number
        if self.output_size == 1:
            x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            x = self.fc(x)
        else:
            # Reshape 2D
            if self.output_channels == 1:
                x = reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x
