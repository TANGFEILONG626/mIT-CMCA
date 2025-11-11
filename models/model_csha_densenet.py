import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # (b, c, 1, 1) -> (b, c)
        y = y.unsqueeze(2)  # (b, c) -> (b, c, 1)
        y = self.conv(y)  # (b, c, 1) -> (b, c, 1)
        y = torch.sigmoid(y).view(b, c, 1, 1)  # (b, c, 1, 1)
        
        return x * y.expand_as(x)  # Apply attention weights to input
    
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # (b, c, 1, 1) -> (b, c)
        y = self.fc(y).view(b, c, 1, 1)     # (b, c) -> (b, c, 1, 1)
        return x * y.expand_as(x)
    
class CSHAAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.eca = ECAAttention(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        eca_out = self.eca(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.spatial(torch.cat([max_pool, avg_pool], dim=1))
        fusion = eca_out * spatial_out.expand_as(eca_out)
        out = fusion + x
        return out
    
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):

        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        return torch.cat([x, new_features], 1)  # Concatenate input with output of the layer


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):

        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        
        self.dense_layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.norm(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class Re_DenseNet121(nn.Module):
    def __init__(self, output_dim=768, growth_rate=32, block_config=(6, 8, 8, 0), compression=1.0, drop_rate=0.0):

        super(Re_DenseNet121, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks
        num_input_features = 64
        self.dense_block1 = _DenseBlock(block_config[0], num_input_features, growth_rate, drop_rate=drop_rate)
        num_input_features += block_config[0] * growth_rate
        self.transition1 = _Transition(num_input_features, int(num_input_features * compression))
        num_input_features = int(num_input_features * compression)

        self.dense_block2 = _DenseBlock(block_config[1], num_input_features, growth_rate, drop_rate=drop_rate)
        num_input_features += block_config[1] * growth_rate
        self.transition2 = _Transition(num_input_features, int(num_input_features * compression))
        num_input_features = int(num_input_features * compression)

        self.dense_block3 = _DenseBlock(block_config[2], num_input_features, growth_rate, drop_rate=drop_rate)
        num_input_features += block_config[2] * growth_rate
        self.transition3 = _Transition(num_input_features, int(num_input_features * compression))
        num_input_features = int(num_input_features * compression)

        self.dense_block4 = _DenseBlock(block_config[3], num_input_features, growth_rate, drop_rate=drop_rate)

        # Final batch norm layer
        self.norm_final = nn.BatchNorm2d(num_input_features)
        self.relu_final = nn.ReLU(inplace=True)

        #self.eca = ECAAttention(num_input_features)
        #self.eca = SEAttention(num_input_features)
        self.eca = CSHAAttention(num_input_features)

        self.fc = nn.Linear(in_features=num_input_features, out_features=output_dim)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.pool1(x)

        x = self.dense_block1(x)
        x = self.transition1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.dense_block3(x)
        x = self.transition3(x)

        x = self.dense_block4(x)

        x = self.relu_final(self.norm_final(x))

        x = self.eca(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  #[batch_size, num_input_features]

        x = self.fc(x)
        return x

def csha_densenet121(output_dim=768):

    return Re_DenseNet121(output_dim)

