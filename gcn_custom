



import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

def load_images_to_graph(dataset_path):
    classes = os.listdir(dataset_path)
    num_classes = len(classes)
    images_per_class = []
    edge_index_list = []
    node_features_list = []
    labels_list = []
    for i, c in enumerate(classes):
        class_path = os.path.join(dataset_path, c)
        images = os.listdir(class_path)
        images_per_class.append(len(images))
        for img in images:
            img_path = os.path.join(class_path, img)
            node_features = np.array(Image.open(img_path).convert('L'))  # convert to grayscale
            height, width = node_features.shape
            x, y = np.meshgrid(range(width), range(height))
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            node_features = np.hstack((x, y, node_features.reshape(-1, 1)))  # (x, y, grayscale) for each pixel
            node_features_list.append(torch.tensor(node_features, dtype=torch.float))
            labels_list.append(i)
            
            # Create edges between neighboring pixels
            edge_index = torch.tensor([], dtype=torch.long)
            for j in range(height):
                for k in range(width):
                    idx = j * width + k
                    if k > 0:
                        edge_index = torch.cat([edge_index, torch.tensor([[idx, idx-1]], dtype=torch.long)], dim=0)
                    if j > 0:
                        edge_index = torch.cat([edge_index, torch.tensor([[idx, idx-width]], dtype=torch.long)], dim=0)
                    if k < width-1:
                        edge_index = torch.cat([edge_index, torch.tensor([[idx, idx+1]], dtype=torch.long)], dim=0)
                    if j < height-1:
                        edge_index = torch.cat([edge_index, torch.tensor([[idx, idx+width]], dtype=torch.long)], dim=0)
            edge_index_list.append(edge_index.t().contiguous())

    # Create PyTorch Geometric Data objects
    data_list = []
    for i in range(len(labels_list)):
        data = Data(x=node_features_list[i], edge_index=edge_index_list[i], y=torch.tensor(labels_list[i], dtype=torch.long))
        data_list.append(data)

    # Create DataLoader
    batch_size = min(images_per_class)  # make sure we have the same number of images per class
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    # Define GCN model
    class GCN(torch.nn.Module):
        def __init__(self, num_features, hidden_size, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_size)
            self.conv2 = GCNConv(hidden_size, num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    # Train GCN model
    model = GCN(num_features=3, hidden_size=16, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        total_loss = 0
    
    
    # Train GCN model
    model = GCN(num_features=3, hidden_size=16, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
    total_loss = 0
      for batch in loader:
          optimizer.zero_grad()
          out = model(batch.x, batch.edge_index)
          loss = F.nll_loss(out, batch.y)
          total_loss += loss.item() * batch.num_graphs
          loss.backward()
          optimizer.step()
      if epoch % 10 == 0:
          print('Epoch {0}, Loss {1}'.format(epoch, total_loss / len(data_list)))






def load_images_to_graph(dataset_path):
    # Create dictionary to map class names to integers
    class_to_idx = {'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3, 'class5': 4, 'class6': 5}

    # Create lists to store node features, edge indices, and class labels for all graphs in dataset
    x_list = []
    edge_index_list = []
    y_list = []

    # Loop over classes in dataset
    for class_name in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, class_name)):
            continue
        class_idx = class_to_idx[class_name]

        # Loop over images in class
        for file_name in os.listdir(os.path.join(dataset_path, class_name)):
            if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
                continue

            # Load image and convert to grayscale
            img_path = os.path.join(dataset_path, class_name, file_name)
            img = Image.open(img_path).convert('L')

            # Convert image to graph and append to lists
            x, edge_index = img_to_graph(img)
            x_list.append(x)
            edge_index_list.append(edge_index)
            y_list.append(class_idx)

    # Create PyTorch Geometric Data object from lists of node features, edge indices, and class labels
    data = Data(x=torch.stack(x_list), edge_index=torch.stack(edge_index_list).t().contiguous(), y=torch.tensor(y_list))

    return data



### GCN FOR RgB imgaes
class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



#GCN for 6 class RGB images
import os
import torch
import torchvision.transforms as transforms
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Define transform for images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to 32x32 pixels
    transforms.ToTensor()  # Convert image to tensor
])

# Define function to convert images to graph-based representation
def img_to_graph_rgb(image_tensor):
    # Extract RGB color channels separately
    red_channel = image_tensor[0, :, :]
    green_channel = image_tensor[1, :, :]
    blue_channel = image_tensor[2, :, :]

    # Convert each color channel to a 1D vector and stack them into a 3D tensor
    nodes = torch.stack([red_channel.view(-1), green_channel.view(-1), blue_channel.view(-1)], dim=1)

    # Create edge indices for a 2D grid graph
    num_pixels = image_tensor.shape[1] * image_tensor.shape[2]
    row_indices = torch.arange(num_pixels)
    col_indices = torch.arange(num_pixels)
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    # Create a PyTorch Geometric Data object
    data = Data(x=nodes, edge_index=edge_index)

    return data

# Define GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define paths to image folders
folder_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']
root_path = '/path/to/parent/folder/'

# Create list of image paths and class labels
image_paths = []
class_labels = []
for i, folder_name in enumerate(folder_names):
    folder_path = os.path.join(root_path, folder_name)
    image_filenames = os.listdir(folder_path)
    for image_filename in image_filenames:
        image_path = os.path.join(folder_path, image_filename)
        image_paths.append(image_path)
        class_labels.append(i)

# Load images and convert them to graph-based representation
data_list = []
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    data = img_to_graph_rgb(image_tensor)
    data_list.append(data)

# Create PyTorch Geometric Data object
loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Train GCN model
model = GCN(num_features=3, hidden_size=16, num_classes=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(200):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    print('Epoch


# Stopped & regerated
model = GCN(num_features=3, hidden_size=16, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    total_loss = 0
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x.float(), batch.edge_index)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, total_loss / len(train_dataset)))

# Evaluate GCN model
model.eval()
correct = 0
for batch in test_loader:
    pred = model(batch.x.float(), batch.edge_index).argmax(dim=1)
    correct += int((pred == batch.y).sum())

print('Accuracy: {:.4f}'.format(correct / len(test_dataset)))

