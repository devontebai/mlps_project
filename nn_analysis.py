# imports
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
torch.manual_seed(42)

class covid_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.dataset.iloc[idx,:-1]
        features = np.array(features)
        targets = self.dataset.iloc[idx, -1].squeeze()
        targets = np.array(targets)
        

        if self.transform:
            sample = self.transform(sample)

        return features, targets

cov_data_train = covid_dataset('tp_wimputed_train.csv', '')
cov_data_test = covid_dataset('tp_wimputed_val.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)
test_loader = DataLoader(cov_data_test, batch_size = 64, shuffle=True)


def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    """
    Train the model for one epoch
 
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader containing the training data
    loss_function : torch.nn.Module
        Loss function to use for optimization
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters
    device : torch.device
        Device to run the training on (CPU or GPU or MPS)
        
    Returns:
    --------
    tuple
        (average_loss, accuracy) for the epoch  
    """    
    # Hint 1: for getting the prediction of the label, check out argmax 
    # function https://pytorch.org/docs/stable/torch.html#reduction-ops

    # Hint 2: If you run into an error when training, print out the dimensions
    # of the output of your neural network and your labels! 

    # (1) SET THE MODEL TO TRAIN
    model.train()

    # initialize variables to collect the loss and accuracy
    running_loss = 0.0
    correct = 0

    for inputs, targets in train_loader:

        # (2) Move data to device and ensure targets are the correct shape
        inputs, targets = inputs.to(device), targets.to(device)
        # (3) Zero the gradients
        optimizer.zero_grad()
       
        # (4) Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()

        #print(targets.shape)
        #pred = torch.argmax(outputs, dim = 1)

        #print(pred)
        # (5) Calculate loss
        loss = loss_function(outputs, targets)
        loss = torch.sqrt(loss)
       
        # (6) Backward pass and take an optimization step
        loss.backward()
        optimizer.step()

        # (7) Calculate the accuracy over the batch

        #num_correct = torch.sum(pred == targets)

        #perc_correct = num_correct/len(train_loader.dataset)
        

        # (8) update the loss tracker and accuracy trackers
        running_loss += loss.item() * inputs.size(0)
        #correct += perc_correct


    # (9) calculate the loss and accuracy over the epoch using our trackers
    epoch_loss = running_loss / len(train_loader.dataset)
    #epoch_accuracy = correct
    
    return epoch_loss

def evaluate(model, test_loader, loss_function, device):
    """
    Evaluate the model on the test dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to evaluate
    test_loader : torch.utils.data.DataLoader
        DataLoader containing the test data
    loss_function : torch.nn.Module
        Loss function to compute the model's performance
    device : torch.device
        Device to run the evaluation on (CPU or GPU or MPS)
        
    Returns:
    --------
    tuple
        (test_loss, test_accuracy) 
    """

    # (1) Set the model to evaluate
    model.eval()
    
    # instantiate variables to collect loss/number of correct items
    running_loss = 0.0
    correct = 0
    
    # (2) use the `no_grad` context-manager to turn off gradient computation, 
    #      which will speed up inference
    # YOUR CODE HERE
    for inputs, targets in test_loader:

        # (3) Move data to device and ensure targets are the correct shape
        inputs, targets = inputs.to(device), targets.to(device)

        # (4) Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze()
        #pred = torch.argmax(outputs, dim = 1)
       
        # (5) Calculate loss
        loss = loss_function(outputs, targets)
        loss = torch.sqrt(loss)

        # (6) Calculate the accuracy over the batch
        #perc_correct = torch.sum(pred == targets)/len(test_loader.dataset)

        # (7) update the loss tracker and accuracy trackers
        running_loss += loss.item() * inputs.size(0)
        #correct += perc_correct

    # (8) calculate the loss and accuracy over the epoch using our trackers
    test_loss = running_loss / len(test_loader.dataset)
    #test_accuracy = correct
    
    return test_loss

def plot_losses(train_losses, test_losses):
    """
    Plot training and test metrics over time
    
    Parameters:
    -----------
    train_losses : list
       List of training loss values per epoch
    test_losses : list
       List of test loss values per epoch
    train_accuracies : list
       List of training accuracy values per epoch
    test_accuracies : list
       List of test accuracy values per epoch
    """
    # Create a figure with two subplots side by side
    fig, (ax1) = plt.subplots(1, figsize=(10, 4))
    
    # Plot 1: Plot losses for each epoch
    # (1) Plot and label the train_losses, test_losses (separately) for each epoch
    # YOUR CODE HERE
    ax1.plot(range(len(train_losses)), train_losses, c='orange', label = 'Training Losses')
    ax1.plot(range(len(test_losses)), test_losses, c = 'green', label = 'Testing Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss Over Time')
    ax1.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network

    A fully connected neural network with two hidden layers of size 128 and ReLU activation

    Parameters:
    -----------
    input_size : int
        Size of the input features (flattened)
    num_classes : int
        Number of output classes
    """
    def __init__(self, input_size, num_classes):
        
        # (1) Initialize the parent class by calling its constructor with super()
        super(MLP, self).__init__()

        # (2) Intialize the MLP layers
        self.hidden_layer1 = nn.Linear(input_size, 128)
        self.hidden_layer2 = nn.Linear(128, 128)
        self.hidden_layer3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, num_classes)
        
        # (3) Intialize the activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, num_classes)
        """
        
        # (1) Reshape input from (batch_size, channels, height, width) to (batch_size, height * width)
        x_reshaped = x.float()
        #print(x_reshaped.shape)

        # (2) Use the reshaped input for the network's forward pass and return final the logits
        out1 = self.hidden_layer1(x_reshaped)
        out1_act = self.activation(out1)
        #print(out1_act.shape)
        out2 = self.hidden_layer2(out1_act)
        out2_act = self.activation(out2)

        out3 = self.hidden_layer3(out2_act)
        out3_act = self.activation(out3)
        outout = self.output_layer(out3_act)
        
        return outout.double()

############################################################
#imputed data with TP target var

cov_data_train = covid_dataset('tp_wimputed_train.csv', '')
cov_data_test = covid_dataset('tp_original_test.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)
test_loader = DataLoader(cov_data_test, batch_size=64, shuffle=True)

# (1) Instantiate the MLP
mlp = MLP(16, 1)
# (2) Instantiate Adam with a learning rate of 0.001
optimizer =  optim.Adam(mlp.parameters(), lr = 0.0001)
# (3) Instantiate the Cross Entropy loss function
criterion =  nn.MSELoss()

# collect the training/test losses/accuracies
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# we will train for 10 epochs only
num_epochs = 50
# if whatever you are running on has gpu support, this will automatically detect it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# cpu may actually be faster for these small batch sizes/simple networks-you can test it out yourself
#device = 'mps' 

# (4) send the model to the device
model = mlp.to(device)

# run the training loop
print(f"Training for {num_epochs} epochs on {device}:")
for epoch in range(num_epochs):

    # (5) train one epoch, get `train_loss, train_accuracy` 
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # (6) evaluate on the test set, get `tst_loss, test_accuraacy`
    test_loss = evaluate(model, test_loader, criterion, device)
    
    # (7) append the losses/accuracies to our lists
    train_losses.append(tr_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Test Loss: {test_loss:.4f}")


plot_losses(train_losses, test_losses)

final_test_loss = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_test_loss:.4f}")

cov_data_train = covid_dataset('tp_wimputed_train.csv', '')
cov_data_test = covid_dataset('tp_original_test.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)
test_loader = DataLoader(cov_data_test, batch_size=64, shuffle=True)



############################################################
#pretraining technique with TP target var

# (1) Instantiate the MLP
mlp = MLP(16, 1)
# (2) Instantiate Adam with a learning rate of 0.001
optimizer =  optim.Adam(mlp.parameters(), lr = 0.0001)
# (3) Instantiate the Cross Entropy loss function
criterion =  nn.MSELoss()

# collect the training/test losses/accuracies
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# we will train for 10 epochs only
num_epochs = 25
# if whatever you are running on has gpu support, this will automatically detect it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# cpu may actually be faster for these small batch sizes/simple networks-you can test it out yourself
#device = 'mps' 

# (4) send the model to the device
model = mlp.to(device)



# run the training loop
print(f"Pre-Training for {num_epochs} epochs on {device}:")
for epoch in range(num_epochs):

    # (5) train one epoch, get `train_loss, train_accuracy` 
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # (6) evaluate on the test set, get `tst_loss, test_accuraacy`
    test_loss = evaluate(model, test_loader, criterion, device)
    
    # (7) append the losses/accuracies to our lists
    train_losses.append(tr_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Test Loss: {test_loss:.4f}")


print('BEGINNING TRAINING')

cov_data_train = covid_dataset('tp_original_train.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)

print(f"Training for {num_epochs} epochs on {device}:")
for epoch in range(num_epochs):

    # (5) train one epoch, get `train_loss, train_accuracy` 
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # (6) evaluate on the test set, get `tst_loss, test_accuraacy`
    test_loss = evaluate(model, test_loader, criterion, device)
    
    # (7) append the losses/accuracies to our lists
    train_losses.append(tr_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Test Loss: {test_loss:.4f}")

plot_losses(train_losses, test_losses)

final_test_loss = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_test_loss:.4f}")


############################################################
#non-imputed data with TP target var
cov_data_train = covid_dataset('tp_original_train.csv', '')
cov_data_val = covid_dataset('tp_original_train.csv', '')
cov_data_test = covid_dataset('tp_original_test.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)
val_loader = DataLoader(cov_data_val, batch_size = 64, shuffle=True)
test_loader = DataLoader(cov_data_test, batch_size=64, shuffle=True)
# (1) Instantiate the MLP
mlp = MLP(16, 1)
# (2) Instantiate Adam with a learning rate of 0.001
optimizer =  optim.Adam(mlp.parameters(), lr = 0.0001)
# (3) Instantiate the Cross Entropy loss function
criterion =  nn.MSELoss()

# collect the training/test losses/accuracies
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# we will train for 10 epochs only
num_epochs = 50
# if whatever you are running on has gpu support, this will automatically detect it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# cpu may actually be faster for these small batch sizes/simple networks-you can test it out yourself
#device = 'mps' 

# (4) send the model to the device
model = mlp.to(device)

# run the training loop
print(f"Training for {num_epochs} epochs on {device}:")
for epoch in range(num_epochs):

    # (5) train one epoch, get `train_loss, train_accuracy` 
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # (6) evaluate on the test set, get `tst_loss, test_accuraacy`
    test_loss = evaluate(model, val_loader, criterion, device)
    
    # (7) append the losses/accuracies to our lists
    train_losses.append(tr_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Test Loss: {test_loss:.4f}")

plot_losses(train_losses, test_losses)

final_test_loss = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_test_loss:.4f}")

############################################################
#cv target var

cov_data_train = covid_dataset('cv_train.csv', '')
cov_data_test = covid_dataset('cv_test.csv', '')
train_loader = DataLoader(cov_data_train, batch_size = 64, shuffle=True)
test_loader = DataLoader(cov_data_test, batch_size=64, shuffle=True)

# (1) Instantiate the MLP
mlp = MLP(16, 1)
# (2) Instantiate Adam with a learning rate of 0.001
optimizer =  optim.Adam(mlp.parameters(), lr = 0.0001)
# (3) Instantiate the Cross Entropy loss function
criterion =  nn.MSELoss()

# collect the training/test losses/accuracies
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# we will train for 10 epochs only
num_epochs = 50
# if whatever you are running on has gpu support, this will automatically detect it
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# cpu may actually be faster for these small batch sizes/simple networks-you can test it out yourself
#device = 'mps' 

# (4) send the model to the device
model = mlp.to(device)

# run the training loop
print(f"Training for {num_epochs} epochs on {device}:")
for epoch in range(num_epochs):

    # (5) train one epoch, get `train_loss, train_accuracy` 
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # (6) evaluate on the test set, get `tst_loss, test_accuraacy`
    test_loss = evaluate(model, test_loader, criterion, device)
    
    # (7) append the losses/accuracies to our lists
    train_losses.append(tr_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Test Loss: {test_loss:.4f}")

plot_losses(train_losses, test_losses)

final_test_loss = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_test_loss:.4f}")