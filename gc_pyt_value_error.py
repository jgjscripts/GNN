import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = torch.mm(x, self.weight)
        return x + self.bias

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (features, adj, labels) in enumerate(loader):
        features, adj, labels = features.to(device), adj.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch: {} | Loss: {:.4f}'.format(epoch, batch_idx, loss.item()))

# Testing function
def test(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, adj, labels in loader:
            features, adj, labels = features.to(device), adj.to(device), labels.to(device)
            output = model(features, adj)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f}%'.format(100 * correct / total))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image tensors
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model parameters
input_dim = 32 * 32 * 3  # Assuming CIFAR-10 images are 32x32 with 3 channels
hidden_dim = 64
output_dim = 10  # Number of CIFAR-10 classes

# Create the GCN model
model = GCN(input_dim, hidden_dim, output_dim).to(device)

# Define optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
