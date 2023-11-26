import torch
from torchvision.transforms import v2 as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
import os
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
import cv2
import numpy as np



#------------------------------------For transforms-----------------------------------------------------------------

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
threshold=.5
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.Grayscale(num_output_channels=3),
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    #lambda x: (x > threshold).float(),  # Apply binary thresholding
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=mean, std=std),
    #lambda x: (x > threshold).float(),  # Apply binary thresholding
])

#------------------------------------------CustomFolder-----------------------------------------------------


class CustomFolder(Dataset):
    def __init__(self, root, transform=None, has_labels=True):
        """
        Custom dataset class for handling images with or without labels.

        Args:
        - root (str): Path to the folder containing images.
        - transform (callable, optional): Optional transform to be applied to each image.
        - has_labels (bool): Indicates whether the folder contains labeled subfolders.
        """
        self.root = root
        self.transform = transform
        self.has_labels = has_labels

        # If labels are available, load them
        if self.has_labels:
            # Get sorted list of class names from folder names
            self.classes = sorted([cls for cls in os.listdir(root) if not cls.startswith('.')])
            # Create a dictionary mapping class names to integer labels
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            # Get a list of image file paths
            self.image_paths = []
            for cls in self.classes:
                cls_folder = os.path.join(root, cls)
                
                cls_image_paths = [os.path.join(cls_folder, img) for img in os.listdir(cls_folder) ]
                self.image_paths.extend(cls_image_paths)
        else :
            # Get a list of image file paths
            if os.path.isdir(root): self.image_paths = [os.path.join(root, img) for img in os.listdir(root) ]
            else :self.image_paths=root
    #def number of of 
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_paths)
    
    def CountPixels(self,img):
        """
        return the 6 additional information of mean and variance 
        """
        mean_color = np.mean(img, axis=(0, 1))/255
        color_std = np.std(img, axis=(0, 1))/255
        # Return a tuple of the original image and the counts
        return np.concatenate((mean_color,color_std))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if type(idx) is str:image_path=self.image_paths
        # Load image from file
        image = Image.open(image_path).convert('RGB')
        counts =self.CountPixels(image)
        if self.transform:
            # Apply optional image transformations
            image = self.transform(self.add_segmentation(image))

        if self.has_labels:
            # Extract label from the folder name
            label = self.class_to_idx[os.path.basename(os.path.dirname(image_path))]
            return image, label, counts,image_path
        else:
            # Placeholder label (can be any value since it won't be used during inference)
            return image, -1, counts,image_path
    def predict_one(self,model):
        """Predict one image"""
        image = Image.open(self.image_paths).convert('RGB')
        counts =torch.tensor(self.CountPixels(image))
        counts= torch.tensor(np.expand_dims(counts, axis=0))
        if self.transform:
            # Apply optional image transformations
            image = self.transform(self.add_segmentation(image))
        image = torch.tensor( np.expand_dims(image, axis=0))
        # Forward pass
        with torch.no_grad():
            outputs = model(image,counts)
        # Convert predicted probabilities to class predictions
            _, predictions = (torch.max(outputs, 1))
            predictions = int(predictions)
            proba=str(float(outputs[0][predictions]))[:6]
            return proba,predictions,['Fields', 'Roads'][predictions]
        

    def visIm(self, num_images=5,random=True,labels=[]):
        """
        Visualizes a random subset of images from the dataset along with their labels.

        Args:
        - num_images (int): Number of images to visualize.
        - random (logic ): to show randomly photos
        - labels (list) : labels are predicted by the model  
        """
        random_indices= [i for i in range(num_images)]
        # Get random indices for images
        if random: random_indices = np.random.choice(len(self), num_images, replace=False)

        # Visualize random images with labels
        for idx in random_indices:
            image, label,_ ,im_path= self[idx]
            
            if labels : label,image=labels[idx],Image.open(im_path).convert('RGB')
            if self.transform: image = Image.open(im_path).convert('RGB')  # Convert from (C, H, W) to (H, W, C)
            # Display the image with label
            im_name=im_path.split('/')[-1] 
            plt.imshow(image)
            if (self.has_labels ): plt.title(f"Label: {label}, Class: {self.classes[label]} - name: {im_name}")
            if (labels ): plt.title(f"prediction: {label}, name: {im_name}")
            plt.show()
        return image
    def counter(self):
        # count labels
        return Counter(label for _, label,_, _ in self)

    def add_segmentation(self,image):
        """
        Roads and fields may have distinct colors. we can apply color segmentation techniques 
        to isolate regions of the image that are likely to be roads or fields based on their color.
        """
        # Convert the image to HSL
        image_hsl = image.convert('HSV')

        # Define the HSV-like ranges for green, yellow, gray, and red
        green_lower = (30, 40, 40)
        green_upper = (80, 255, 255)

        yellow_lower = (20, 100, 100)
        yellow_upper = (40, 255, 255)

        gray_lower = (0, 0, 40)
        gray_upper = (180, 30, 220)

        # Red is a bit tricky due to the circular nature of the HSL color space
        red_lower1 = (0, 100, 100)
        red_upper1 = (10, 255, 255)

        red_lower2 = (160, 100, 100)
        red_upper2 = (180, 255, 255)

        # Convert image to NumPy array for manipulation
        image_array = np.array(image_hsl)

        # Create masks based on the color ranges
        green_mask = np.all((image_array >= green_lower) & (image_array <= green_upper), axis=-1)
        yellow_mask = np.all((image_array >= yellow_lower) & (image_array <= yellow_upper), axis=-1)
        gray_mask = np.all((image_array >= gray_lower) & (image_array <= gray_upper), axis=-1)
        red_mask1 = np.all((image_array >= red_lower1) & (image_array <= red_upper1), axis=-1)
        red_mask2 = np.all((image_array >= red_lower2) & (image_array <= red_upper2), axis=-1)

        # Combine the masks
        combined_mask = green_mask | yellow_mask | gray_mask | red_mask1 | red_mask2

        # Apply the combined mask to the original image
        segmented_image_array = image_array.copy()
        segmented_image_array[~combined_mask] = 0

        # Convert NumPy array back to Pillow image
        segmented_image = Image.fromarray(segmented_image_array, 'HSV')

        # Convert the segmented image back to RGB
        segmented_image_rgb = segmented_image.convert('RGB')
        return segmented_image_rgb


    def label_distribution(self):
        """
        Calculates and displays the distribution of labels in the dataset.
        """
        if self.has_labels:
            # Get the counts of each label using Counter
            label_counts =self.counter()
            # Plot the label distribution
            plt.bar(self.classes, (label_counts.values()))
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("Label Distribution")
            # Display the count values on top of each bar
            for label, count in label_counts.items():
                plt.text(label, count , str(count), ha='center', va='top')
            plt.show()
        else:
            print("No labels available for distribution calculation.")
    def train_loader(self,dataset,batch_size=8,transform=None):
        """loader for training, by dividing the data into train, valid and test """
        #
        train_size = int(0.6 * len(dataset))
        valid_size = int(0.2 * len(dataset))  
        generator1 = torch.Generator().manual_seed(0)
        test_size  = len(dataset) - (train_size + valid_size)
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset,[train_size, valid_size, test_size],generator=generator1)
        
        # Apply specific transformations to each split
        train_dataset.dataset.transform = train_transform
        valid_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform
        # Calculate class weights for balanced sampling
        class_counts =train_dataset.dataset.counter()
        num_classes = len(class_counts)
        class_weights = 1.0 / np.array([class_counts[class_idx] for class_idx in range(num_classes)])
        # Compute class weights as the inverse of the number of samples for each class
        sample_weights = [class_weights[class_idx[1]] for class_idx in train_dataset]
        # Use WeightedRandomSampler to balance the dataset during training
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size )
        return train_loader,val_loader,test_loader#,dataset


    def predction_loader(self,predict_dataset ,batch_size=8):
        """loader for predicion """
-        predict_loader= DataLoader(predict_dataset, batch_size=batch_size )
        return predict_loader



#------------------------------------------Metrics-----------------------------------------------------
  

def met(model,test_loader,show_det=False):
    """ Metrics and Prediction"""
    criterion = torch.nn.CrossEntropyLoss()
    # Testing loop
    # Set the model to evaluation mode
    model.eval()
    # Lists to store predictions and true labels
    all_test_predictions = []
    all_test_labels = []
    running_loss=0
    # DataLoader for the test set
    # Iterate through the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, labels, cou,image_path in test_loader:
            inputs, labels ,cou= inputs.to(device), labels.to(device),cou.to(device)  # Move data to GPU
            # Forward pass
            outputs = model(inputs,cou)
            if labels[0] !=-1:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            # Convert predicted probabilities to class predictions
            _, predictions = torch.max(outputs, 1)
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy()) 
    # Calculate metrics for the test set
    if labels[0] ==-1: return all_test_predictions
    print("Test loss: ",running_loss)
    f1 = f1_score(all_test_labels, all_test_predictions, average='weighted')
    if show_det:
        test_report = classification_report(all_test_labels, all_test_predictions)
        test_cm = confusion_matrix(all_test_labels, all_test_predictions)
        # Print classification report and confusion matrix for the test set
        print("Test Classification Report:")
        print(test_report)
        # Plot confusion matrix for the test set
        plt.figure(figsize=(5, 5))
        sns.heatmap(test_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    else : 
        accuracy = accuracy_score(all_test_labels, all_test_predictions)
        print(f"Test- Accuracy: {accuracy}, F1 Score: {f1}")
        print("--------------------------------------------------------------------------")
    return outputs,f1,running_loss
#------------------------------------------diagnostic-----------------------------------------------------

def diagnostic(model,test_loader,device=None,):
    """Diagnostic the results of mislabeling """
    # Lists to store misclassified examples
    misclassified_images = []
    misclassified_predictions = []
    misclassified_labels = []
    # Iterate through the test set
    with torch.no_grad():
         for inputs, labels, cou ,image_path in test_loader:
            inputs, labels ,cou= inputs.to(device), labels.to(device),cou.to(device)  # Move data to GPU
            # Forward pass
            outputs = model(inputs,cou)
            # Convert predicted probabilities to class predictions
            _, predictions = torch.max(outputs, 1)

            # Check for misclassified examples
            misclassified_mask = predictions != labels

            if misclassified_mask.any():
                misclassified_images.extend(np.array(image_path)[misclassified_mask])
                misclassified_predictions.extend(predictions[misclassified_mask].cpu().numpy())
                misclassified_labels.extend(labels[misclassified_mask].cpu().numpy())

    # Visualize some of the misclassified examples
    for i in range(min(5, len(misclassified_images))):
        #image = misclassified_images[i].transpose((1, 2, 0)) 
        im_path=misclassified_images[i]
        image =  Image.open(im_path).convert('RGB')
        true_label = misclassified_labels[i]
        predicted_label = misclassified_predictions[i]
        im_name=im_path.split('/')[-1] 
        # Save or display the misclassified image
        #image.save(os.path.join(save_dir, f"misclassified_{i}_true_{true_label}_pred_{predicted_label}.png"))
        plt.imshow(image )
        names=['Fields', 'Roads']
        plt.title(f"True: {names[true_label]}, Predicted: {names[predicted_label]} - name: {im_name}")
        plt.show()

#------------------------------------------loss-----------------------------------------------------


def plt_loss(train_losses,test_losses,epochs):
    """to plot losses """
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.show()