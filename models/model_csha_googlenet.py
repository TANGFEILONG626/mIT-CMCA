import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, negative_slope=0.1, inplace=True)

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
        
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (b, 1, h, w)
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # (b, 1, h, w)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)  # (b, 2, h, w)
        spatial_weights = self.spatial(spatial_input)     # (b, 1, h, w)
        
        channel_attended = eca_out
        spatial_attended = x * spatial_weights
        
        out = channel_attended + spatial_attended + x
        
        return out

class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = max(3, t if t % 2 else t + 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg_pool(x)  # (b, c, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)  # (b, 1, c)

        y = self.conv(y)  # (b, 1, c)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)  # (b, c, 1, 1)
        
        return x * y.expand_as(x)

class ResEInceptionModule(nn.Module):
    def __init__(self, in_channels, conv1_out, conv3_reduce, conv3_out, conv5_reduce, conv5_out, pool_proj):
        super(ResEInceptionModule, self).__init__()
        self.branch1 = BasicConv2d(in_channels, conv1_out, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, conv3_reduce, kernel_size=1),
            BasicConv2d(conv3_reduce, conv3_out, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, conv5_reduce, kernel_size=1),
            BasicConv2d(conv5_reduce, conv5_out, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
        #self.eca = ECAAttention(conv1_out + conv3_out + conv5_out + pool_proj)
        #self.eca = SEAttention(conv1_out + conv3_out + conv5_out + pool_proj)
        self.eca = CSHAAttention(conv1_out + conv3_out + conv5_out + pool_proj)
        self.residual_conv = BasicConv2d(in_channels, conv1_out + conv3_out + conv5_out + pool_proj, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.eca(out)
        return F.relu(out + residual)

class rEGoogLeNet(nn.Module):
    def __init__(self, output_dim=768):
        super(rEGoogLeNet, self).__init__()

        self.stem = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = ResEInceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = ResEInceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = ResEInceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = ResEInceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = ResEInceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = ResEInceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = ResEInceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception5a = ResEInceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = ResEInceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(in_features=1024, out_features=output_dim)

    def forward(self, x):
        x = self.stem(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

def csha_googlenet(output_dim=1024):
    return rEGoogLeNet(output_dim=output_dim)
