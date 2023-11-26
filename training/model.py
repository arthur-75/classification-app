import torch
import copy
from torchvision.models import resnet18,ResNet18_Weights, googlenet,GoogLeNet_Weights
import torch.optim as optim
import torch.nn as nn
from training.utils import met
import torch.nn.functional as F
from datetime import datetime


class Classifier(nn.Module):
    """
    The main model to classify
    """
    def __init__(self, num_classes,pretrained=True):
        super(Classifier, self).__init__()
        # Use a pre-trained  model as an examples
        self.pretrained = googlenet(weights=GoogLeNet_Weights.DEFAULT)#resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze the weights of pretrained layers
        for param in self.pretrained.parameters():
            param.requires_grad = False
        # Modify the last fully connected layer for the number of output classes
        in_features = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(in_features, 300)
        self.dropout1 = nn.Dropout(0.3)  # Fix the typo here
        self.dropout2 = nn.Dropout(0.3)  # Fix the typo here
        self.f1 = nn.Linear(300, 50)
        self.m1 = nn.Linear(6, 200)
        self.f2 = nn.Linear(250, num_classes)
    def forward(self, x,counts):
        counts = counts.to(x.dtype)  # Ensure that counts has the same dtype as x
        img = x
        x = F.relu(self.pretrained(img))
        x = self.dropout1(x)
        x = F.relu(self.f1(x))
        counts = F.relu(self.m1(counts))
        x = torch.cat([x, counts], dim=1)  # Concatenate counts to the output of pretrained
        x = self.dropout2(x)  # Corrected the dropout layer
        x = F.softmax(self.f2(x),dim=1) 
        return x
    
def training(model,train_loader,val_loader,epochs=15):
    """Training a model""" 
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the GPU
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Create the Adam optimizer

    # Set up the ExponentialLR scheduler
    scheduler =optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9 )

    # Training loop
    best_valid_f1=0.0
    best_valid_loss=1000.0
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels, cou,_ in train_loader:
            inputs, labels ,cou= inputs.to(device), labels.to(device),(cou ).to(device)  # Move data to GPU
            optimizer.zero_grad()  # Zero the gradients
            # Forward pass
            outputs = model(inputs,cou)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Print average training loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        train_losses.append(epoch_loss)
        outputs,f1,test_lo=met(model,val_loader,False)
        test_losses.append(test_lo / len(val_loader))
        # Check if the current model has the best F1 score
        scheduler.step()
        if( f1 >= best_valid_f1) and ( test_lo <= best_valid_loss):
            
            print("Found best Model")
            print("#####################################")
            best_valid_f1 = f1
            best_valid_loss = test_lo
            best_model=copy.deepcopy(model)
            # Save the model
    torch.save(best_model.state_dict(), f'training/w_models/{str(datetime.today())}_model.pth')
    return best_model,train_losses ,test_losses




def model():
    """Use a trained model that I have built"""
    state_dict = torch.load("training/w_models/best_model.pth", map_location=torch.device('cpu'))
    model = Classifier(2)
    model_dict = model.state_dict()
    # Map the weights from the pre-trained model to your model
    pretrained_dict = {k: v for k, v in zip(model_dict.keys(),state_dict.values())}
    model_dict.update(pretrained_dict)
    # Load the updated state dictionary into your model
    model.load_state_dict(model_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ev=model.eval()
    return model