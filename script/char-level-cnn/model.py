#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.args = args

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=args.num_features, out_channels=256,
                      kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(1024, 4)

        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        if self.args.debug:
            print("conv1 output", x.size())
        x = self.conv2(x)
        if self.args.debug:
            print('conv2 output', x.size())

        x = self.conv3(x)
        if self.args.debug:
            print('conv3 ouptut', x.size())

        x = self.conv4(x)
        if self.args.debug:
            print('conv4 output', x.size())

        x = self.conv5(x)
        if self.args.debug:
            print('conv5 output', x.size())

        x = self.conv6(x)
        if self.args.debug:
            print('conv6 ouput', x.size())

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.fc2(x)

        x =self.fc3(x)

        x = self.log_softmax(x)

        return x


