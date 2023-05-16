## VGG 16 GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias

# Modify the VGG16 model for graph-based tasks
class VGG16_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VGG16_GNN, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Example usage
input_dim = 512  # Dimensionality of VGG16 output
hidden_dim = 256
output_dim = 10  # Number of classes

# Dummy input and adjacency matrix (replace with actual data)
x = torch.randn(1, input_dim)
adj = torch.randn(1, input_dim, input_dim)

model = VGG16_GNN(input_dim, hidden_dim, output_dim)
output = model(x, adj)
print(output)




##############################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias

# Define the VGG16 model
class VGG16_GCN(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_GCN, self).__init__()
        self.conv1_1 = GraphConvolution(3, 64)
        self.conv1_2 = GraphConvolution(64, 64)
        self.conv2_1 = GraphConvolution(64, 128)
        self.conv2_2 = GraphConvolution(128, 128)
        self.conv3_1 = GraphConvolution(128, 256)
        self.conv3_2 = GraphConvolution(256, 256)
        self.conv3_3 = GraphConvolution(256, 256)
        self.conv4_1 = GraphConvolution(256, 512)
        self.conv4_2 = GraphConvolution(512, 512)
        self.conv4_3 = GraphConvolution(512, 512)
        self.conv5_1 = GraphConvolution(512, 512)
        self.conv5_2 = GraphConvolution(512, 512)
        self.conv5_3 = GraphConvolution(512, 512)
        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.conv1_1(x, adj))
        x = F.relu(self.conv1_2(x, adj))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_1(x, adj))
        x = F.relu(self.conv2_2(x, adj))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3_1(x, adj))
        x = F.relu(self.conv3_2(x, adj))
        x = F.relu(self.conv3_3(x, adj))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_1(x, adj))
        x = F.relu(self.conv4_2(x, adj))
        x = F.relu(self.conv4_3(x, adj))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5_1(x, adj))
        x = F.relu(self.conv5_2(x, adj))
        x = F.relu(self.conv5_3(x, adj))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
	return F.log_softmax(x, dim=1)

# Example usage
num_classes = 10  # Number of classes

# Dummy input and adjacency matrix (replace with actual data)
x = torch.randn(1, 3, 32, 32)
adj = torch.randn(1, 512, 512)

model = VGG16_GCN(num_classes)
output = model(x, adj)
print(output)


#################################
RESNET GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias

# Define the ResNet model modified for graph-based tasks
class ResNet_GCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_GCN, self).__init__()
        self.resnet = resnet.resnet18(pretrained=False)
        self.conv1 = GraphConvolution(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.conv1(x, adj)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Example usage
num_classes = 10  # Number of classes

# Dummy input and adjacency matrix (replace with actual data)
x = torch.randn(1, 3, 224, 224)
adj = torch.randn(1, 512, 512)

model = ResNet_GCN(num_classes)
output = model(x, adj)
print(output)


##############################################################
# RESNET based from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias

# Residual block for GCN
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        residual = x
        out = self.relu(self.conv1(x, adj))
        out = self.conv2(out, adj)
        out += residual
        out = self.relu(out)
        return out

# Define the ResNet_GCN model
class ResNet_GCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_GCN, self).__init__()
        self.conv1 = GraphConvolution(3, 64)
        self.residual_block1 = ResidualBlock(64, 64, 64)
        self.residual_block2 = ResidualBlock(64, 64, 64)
        self.residual_block3 = ResidualBlock(64, 64, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.residual_block1(x, adj)
        x = self.residual_block2(x, adj)
        x = self.residual_block3(x, adj)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage
num_classes = 10  # Number of classes

# Dummy input and adjacency matrix (replace with actual data)
x = torch.randn(1, 3, 32, 32)
adj = torch.randn(1, 32, 32)

model = ResNet_GCN(num_classes)
output = model(x, adj)
print(output)


##############################################################
## Renet18 from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = torch.matmul(x, self.weight)
        return x + self.bias

# Basic block for GCN
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = GraphConvolution(input_dim, output_dim)
        self.conv2 = GraphConvolution(output_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, adj):
        residual = x
        out = self.relu(self.conv1(x, adj))
        out = self.conv2(out, adj)
        if self.stride != 1 or x.shape[1] != out.shape[1]:
            residual = self.conv1(x, adj)
        out += residual
        out = self.relu(out)
        return out

# Define the ResNet_GCN model
class ResNet_GCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_GCN, self).__init__()
        self.conv1 = GraphConvolution(3, 64)
        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, input_dim, output_dim, num_blocks, stride=1):
        layers = []
        layers.append(BasicBlock(input_dim, output_dim, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(output_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        x = self.layer4(x, adj)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage
num_classes = 10  # Number of classes

# Dummy input and adjacency matrix (replace with actual data)
x = torch.randn(1, 3, 32, 32)
adj = torch.randn(1, 32, 32)

model = ResNet_GCN(num_classes)
output = model(x, adj)
print(output)

