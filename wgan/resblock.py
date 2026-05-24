from torch import nn

class CondBN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        
        self.mean_embedding = nn.Embedding(num_embeddings = num_classes, embedding_dim = num_features)
        self.logvar_embedding = nn.Embedding(num_embeddings = num_classes, embedding_dim = num_features)

        self.mean_embedding.weight.data.fill_(0)
        self.logvar_embedding.weight.data.fill_(1)

        self.bn = nn.BatchNorm2d(num_features = num_features, affine = False)

    def forward(self, x, c):
        mean_emb, var_emb = self.mean_embedding(c)[:, :, None, None], self.logvar_embedding(c)[:, :, None, None]
        x = self.bn(x)
        x = mean_emb + x * var_emb
        return x

class GeneratorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, upsample = False):
        super().__init__()

        if upsample == False:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.upsample = lambda x: x
        else:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'nearest'),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            )
            self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')

        self.norm1 = CondBN(in_channels, num_classes)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = CondBN(out_channels, num_classes)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
    def forward(self, x, c):
        s = self.skip(x)
        
        x = self.norm1(x, c)
        x = self.act1(x)
        x = self.upsample(x)
        x = self.conv1(x)
        
        x = self.norm2(x, c)
        x = self.act2(x)
        x = self.conv2(x)
        
        x = x + s
        return x

class CriticResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_size, downsample = False):
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

        self.norm1 = nn.LayerNorm((in_channels, *norm_size))
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.LayerNorm((out_channels, *norm_size))
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
