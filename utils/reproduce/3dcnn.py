import numpy as np
import torch.nn as nn
import argparse
import os


class CNN3D_LBA(nn.Module):
    def __init__(self, in_channels, spatial_size, conv_drop_rate, fc_drop_rate, conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides, fc_units, batch_norm=True, dropout=False):
        super(CNN3D_LBA, self).__init__()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))

        # Convs
        for i in range(len(conv_filters)):
            layers.extend([nn.Conv3d(in_channels, conv_filters[i], kernel_size=conv_kernel_size, bias=True), nn.ReLU()])
            spatial_size -= (conv_kernel_size - 1)
            if max_pool_positions[i]:
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i] - 1) - 1) / max_pool_strides[i] + 1))
            if batch_norm:
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size ** 3)
        # FC layers
        for units in fc_units:
            layers.extend([nn.Linear(in_features, units), nn.ReLU()])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)


# Construct model
def conv_model(in_channels, spatial_size, args):
    num_conv = args.num_conv
    conv_filters = [32 * (2 ** n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1] * int((num_conv + 1) / 2)
    max_pool_sizes = [2] * num_conv
    max_pool_strides = [2] * num_conv
    fc_units = [512]

    model = CNN3D_LBA(in_channels, spatial_size, args.conv_drop_rate, args.fc_drop_rate, conv_filters, conv_kernel_size,
        max_pool_positions, max_pool_sizes, max_pool_strides, fc_units, batch_norm=args.batch_norm,
        dropout=not args.no_dropout)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--unobserved', action='store_true', default=False)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--conv_drop_rate', type=float, default=0.1)
    parser.add_argument('--fc_drop_rate', type=float, default=0.25)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--num_conv', type=int, default=4)
    parser.add_argument('--batch_norm', action='store_true', default=False)
    parser.add_argument('--no_dropout', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))

    args = parser.parse_args()
    model = conv_model(3, 20, args)
    print(sum(p.numel() for p in model.parameters()))
