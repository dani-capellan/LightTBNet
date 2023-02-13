import torch
from torch import nn

class CustomResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        # Constructor
        super().__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(in_channels,out_channels[0],kernel_size=3,stride=1,padding='same')
        self.conv2 = nn.Conv2d(out_channels[0],out_channels[1],kernel_size=3,stride=1,padding='same')
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels[1],kernel_size=1,stride=1,padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels[1])
        )
        # Batch Norms
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        # ReLU
        self.relu = nn.ReLU()

    def forward(self,input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.relu(self.conv1(input)))
        input = self.bn2(self.relu(self.conv2(input)))
        # input = input+shortcut
        input = torch.cat([input,shortcut],dim=1)
        # return self.relu(input)
        return input

class LightTBNet_3blocks(nn.Module):
    def __init__(self, in_channels, CustomResBlock, outputs):
        # Constructor
        super().__init__()
        # Layers

        layer0 = nn.Sequential(
            CustomResBlock(in_channels,[16,16]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
        )
        layer1 = nn.Sequential(
            CustomResBlock(32,[32,32]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        layer2 = nn.Sequential(
            CustomResBlock(64,[64,64]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
        )
        layer3 = nn.Sequential(
            nn.Conv2d(128,out_channels=16,kernel_size=1,padding='valid'),
            nn.ReLU(),
            # nn.BatchNorm2d(16)
        )

        self.features = nn.Sequential()
        self.features.add_module('layer0',layer0)
        self.features.add_module('layer1',layer1)
        self.features.add_module('layer2',layer2)
        self.features.add_module('layer3',layer3)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*16,256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256,outputs),
            # nn.Softmax()  # Included in the Loss
        )

    def forward(self,input):
        input = self.features(input)
        input = self.classifier(input)

        return input

class LightTBNet_4blocks(nn.Module):
    def __init__(self, in_channels, CustomResBlock, outputs):
        # Constructor
        super().__init__()
        # Layers

        layer0 = nn.Sequential(
            CustomResBlock(in_channels,[16,16]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
        )
        layer1 = nn.Sequential(
            CustomResBlock(32,[32,32]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        layer2 = nn.Sequential(
            CustomResBlock(64,[64,64]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
        )
        layer3 = nn.Sequential(
            CustomResBlock(128,[128,128]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
        )
        layer4 = nn.Sequential(
            nn.Conv2d(256,out_channels=16,kernel_size=1,padding='valid'),
            nn.ReLU(),
            # nn.BatchNorm2d(16)
        )

        self.features = nn.Sequential()
        self.features.add_module('layer0',layer0)
        self.features.add_module('layer1',layer1)
        self.features.add_module('layer2',layer2)
        self.features.add_module('layer3',layer3)
        self.features.add_module('layer4',layer4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16*16,256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256,outputs),
            # nn.Softmax()  # Included in the Loss
        )

    def forward(self,input):
        input = self.features(input)
        input = self.classifier(input)

        return input

class LightTBNet_5blocks(nn.Module):
    def __init__(self, in_channels, CustomResBlock, outputs):
        # Constructor
        super().__init__()
        # Layers

        layer0 = nn.Sequential(
            CustomResBlock(in_channels,[16,16]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
        )
        layer1 = nn.Sequential(
            CustomResBlock(32,[32,32]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
        )
        layer2 = nn.Sequential(
            CustomResBlock(64,[64,64]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
        )
        layer3 = nn.Sequential(
            CustomResBlock(128,[128,128]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
        )
        layer4 = nn.Sequential(
            CustomResBlock(256,[256,256]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(512),
        )
        layer5 = nn.Sequential(
            nn.Conv2d(512,out_channels=16,kernel_size=1,padding='valid'),
            nn.ReLU(),
            # nn.BatchNorm2d(16)
        )

        self.features = nn.Sequential()
        self.features.add_module('layer0',layer0)
        self.features.add_module('layer1',layer1)
        self.features.add_module('layer2',layer2)
        self.features.add_module('layer3',layer3)
        self.features.add_module('layer4',layer4)
        self.features.add_module('layer5',layer5)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*16,256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256,outputs),
            # nn.Softmax()  # Included in the Loss
        )

    def forward(self,input):
        input = self.features(input)
        input = self.classifier(input)

        return input