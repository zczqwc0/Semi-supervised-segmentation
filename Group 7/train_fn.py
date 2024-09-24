import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from linknet import link_net
from train_helper import semi_supervised_bce_loss, iou_score, dice_score, precision, recall, specificity
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_labeled_and_unlabeled(train_loader_with_label, train_loader_without_label, val_loader, device, num_epochs=50, lr=1e-4, csv_filename="evaluation_metrics.csv"):
    """
    Train a semi-supervised segmentation model with labeled and unlabeled data.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        train_loader_without_label (DataLoader): Unlabeled training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Default is 50.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        csv_filename (string, optional): Name of the .csv file containing validation results. Default is "evaluation_metrics.csv"

    Returns:
        nn.Module: The trained segmentation model.
    """
    # Initialize the neural network
    model = link_net(classes=1).to(device)    

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training with BCE loss function..")
    
    # Create the val_results folder if it doesn't exist
    os.makedirs('val_results', exist_ok=True)
    
    # Create CSV file to store evaluation metrics in the specified folder
    with open(os.path.join('val_results', csv_filename), mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Epoch', 'Validation Loss', 'IoU Score', 'Dice Score', 'Precision', 'Recall', 'Specificity'])

        for epoch in range(num_epochs):
            running_loss = 0.0
            running_labeled_loss = 0.0
            running_unlabeled_loss = 0.0

            # Set alpha based on epoch
            t1 = 10
            t2 = 60
            alpha_f = 3

            if epoch < t1:
                alpha = 0
            elif epoch < t2:
                alpha = (epoch - t1) / (t2 - t1) * alpha_f
            else:
                alpha = alpha_f

            # Train on both labeled and unlabeled data during each epoch of training
            train_iter_without_label = iter(train_loader_without_label)
            for i, (images_with_label, labels) in enumerate(train_loader_with_label):
                try:
                    images_without_label, _ = next(train_iter_without_label)
                except StopIteration:
                    train_iter_without_label = iter(train_loader_without_label)
                    images_without_label, _ = next(train_iter_without_label)

                images_with_label, labels = images_with_label.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                images_without_label = images_without_label.to(device, dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                pred_with_label = model(images_with_label)
                pred_without_label = model(images_without_label)

                loss, labeled_loss, unlabeled_loss = semi_supervised_bce_loss(pred_with_label, labels, pred_without_label, alpha=alpha)

                # backward propagation
                loss.backward()

                # optimise
                optimizer.step()

                # # print stats every iteration
                # print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f}, labeled loss = {labeled_loss.item():.6f}, unlabeled loss = {unlabeled_loss.item():.6f}, alpha = {alpha}")

                # # print statistics every 10 iteratrions
                # running_loss += loss.item()
                # running_labeled_loss += labeled_loss.item()
                # running_unlabeled_loss += unlabeled_loss.item()
                # if i % 10 == 9:
                #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 10:.6f},  labeled loss = {running_labeled_loss / 10:.6f}, unlabeled loss = {running_unlabeled_loss / 10:.6f}, alpha = {alpha}")
                #     running_loss = 0.0
                #     running_labeled_loss = 0.0
                #     running_unlabeled_loss = 0.0
                    
            # Evaluate the model on the validation set
            model.eval()

            val_loss = 0.0
            total_iou_score = 0.0
            total_dice_score = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_specificity = 0.0

            criterion = nn.BCEWithLogitsLoss()

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
                    output = model(data)

                    loss = criterion(output, target)
                    val_loss += loss.item()
                    total_iou_score += iou_score(output, target)
                    total_dice_score += dice_score(output, target)
                    total_precision += precision(output, target)
                    total_recall += recall(output, target)
                    total_specificity += specificity(output, target)

            print(f"After Epoch {epoch+1}: Validation loss = {(val_loss/len(val_loader)):.6f}, IoU Score = {(total_iou_score/len(val_loader)):.6f}, Dice Score = {(total_dice_score/len(val_loader)):.6f}, Precision = {(total_precision/len(val_loader)):.6f}, Recall = {(total_recall/len(val_loader)):.6f}, Specificity = {(total_specificity/len(val_loader)):.6f}")
            
            # Write evaluation metrics to CSV file
            csv_writer.writerow([epoch + 1, val_loss / len(val_loader), total_iou_score / len(val_loader), total_dice_score / len(val_loader), total_precision / len(val_loader), total_recall / len(val_loader), total_specificity / len(val_loader)])
                
    print("Training completed.")

    return model

def train_labeled_only(train_loader_with_label, val_loader, device, num_epochs=50, lr=1e-4, csv_filename="evaluation_metrics.csv"):
    """
    Train a supervised segmentation model with labeled data only.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Default is 50.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.

    Returns:
        nn.Module: The trained segmentation model.
    """
    # Initialize the neural network
    model = link_net(classes=1).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training with BCE loss function..")
    
    # Create the val_results folder if it doesn't exist
    os.makedirs('val_results', exist_ok=True)
    
    # Create CSV file to store evaluation metrics in the specified folder
    with open(os.path.join('val_results', csv_filename), mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Epoch', 'Validation Loss', 'IoU Score', 'Dice Score', 'Precision', 'Recall', 'Specificity'])

        for epoch in range(num_epochs):
            running_loss = 0.0

            # Train on labeled data only
            for i, (images_with_label, labels) in enumerate(train_loader_with_label):
                images_with_label, labels = images_with_label.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                pred_with_label = model(images_with_label)

                loss = criterion(pred_with_label, labels)

                loss.backward()
                optimizer.step()

                # # print stats every iteration
                # print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f}")

                # # print statistics every 10 iteratrions
                # running_loss += loss.item()
                # if i % 10 == 9:
                #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 10:.6f}")
                #     running_loss = 0.0

            # Evaluate the model on the validation set
            model.eval()

            val_loss = 0.0
            total_iou_score = 0.0
            total_dice_score = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_specificity = 0.0

            criterion = nn.BCEWithLogitsLoss()

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
                    output = model(data)

                    loss = criterion(output, target)
                    val_loss += loss.item()
                    total_iou_score += iou_score(output, target)
                    total_dice_score += dice_score(output, target)
                    total_precision += precision(output, target)
                    total_recall += recall(output, target)
                    total_specificity += specificity(output, target)

            print(f"After Epoch {epoch+1}: Validation loss = {(val_loss/len(val_loader)):.6f}, IoU Score = {(total_iou_score/len(val_loader)):.6f}, Dice Score = {(total_dice_score/len(val_loader)):.6f}, Precision = {(total_precision/len(val_loader)):.6f}, Recall = {(total_recall/len(val_loader)):.6f}, Specificity = {(total_specificity/len(val_loader)):.6f}")
            
            # Write evaluation metrics to CSV file
            csv_writer.writerow([epoch + 1, val_loss / len(val_loader), total_iou_score / len(val_loader), total_dice_score / len(val_loader), total_precision / len(val_loader), total_recall / len(val_loader), total_specificity / len(val_loader)])
                
    print("Training completed.")

    return model

if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create some dummy datasets for testing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    
    # Assume train_labeled, train_unlabeled, val sets for testing purposes
    train_labeled_dataset = datasets.FakeData(transform=transform)  # Replace with actual labeled dataset
    train_unlabeled_dataset = datasets.FakeData(transform=transform)  # Replace with actual unlabeled dataset
    val_dataset = datasets.FakeData(transform=transform)  # Replace with actual validation dataset
    
    # Create DataLoaders for testing
    batch_size = 4
    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Test the semi-supervised training function
    print("Testing semi-supervised training with labeled and unlabeled data...")
    trained_model_semi_supervised = train_labeled_and_unlabeled(
        train_loader_with_label=train_labeled_loader,
        train_loader_without_label=train_unlabeled_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,  # For testing, we can use a smaller number of epochs
        lr=1e-4,
        csv_filename="test_semi_supervised_metrics.csv"
    )

    # Test the supervised training function
    print("Testing supervised training with labeled data only...")
    trained_model_supervised = train_labeled_only(
        train_loader_with_label=train_labeled_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2,  # For testing, use fewer epochs
        lr=1e-4,
        csv_filename="test_supervised_metrics.csv"
    )

    print("Testing complete.")
