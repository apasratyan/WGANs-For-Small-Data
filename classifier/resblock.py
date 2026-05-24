from torch import nn

class ClassifierResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super().__init__()

        if downsample == False:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.downsample = lambda x: x
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.AvgPool2d(kernel_size = 2)
            )
            self.downsample = nn.AvgPool2d(kernel_size = 2)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
    def forward(self, x):
        s = self.skip(x)
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.downsample(x)
        
        x = x + s
        return x
