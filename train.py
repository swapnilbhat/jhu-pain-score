import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from dataloaders import prepare_data
from model import DualXRayNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import os

def train_model(model, train_loader, val_loader,test_loader, num_epochs=50, device='cuda'):
    """
    Train the dual-input X-ray model
    """
    criterion = nn.MSELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Create a timestamped directory for saving models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to {save_dir}")
    
    best_val_loss = float('inf')
    best_model_state = None
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for front_imgs, side_imgs, scores in pbar:
            # Move data to device
            front_imgs = front_imgs.to(device)
            side_imgs = side_imgs.to(device)
            scores = scores.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(front_imgs, side_imgs)
#             print('outputs',outputs)
            loss = criterion(outputs.squeeze(), scores)
#             print('loss',loss)
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'train_loss': np.mean(train_losses)})
            
        train_loss=np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for front_imgs, side_imgs, scores in val_loader:
                front_imgs = front_imgs.to(device)
                side_imgs = side_imgs.to(device)
                scores = scores.to(device)
                
                outputs = model(front_imgs, side_imgs)
                loss = criterion(outputs.squeeze(), scores)
                
                val_losses.append(loss.item())
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_true.extend(scores.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        val_mse = mean_squared_error(val_true, val_preds)
        val_mae = mean_absolute_error(val_true, val_preds)
        
        # Save a checkpoint after every epoch
        if (epoch+1)%5==0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Train Loss: {np.mean(train_losses):.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss) #train_loss is used in the only classifier training
        evaluate_model(model,test_loader,device)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(best_model_state, best_model_path)
            print(f"Best model saved to {best_model_path}")
            
    # Load best model state
    model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test set
    """
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for front_imgs, side_imgs, scores in test_loader:
            front_imgs = front_imgs.to(device)
            side_imgs = side_imgs.to(device)
            
            outputs = model(front_imgs, side_imgs)

            test_preds.extend(outputs.squeeze().cpu().numpy())  # Safely convert to list
            test_true.extend(scores.numpy())
    
    # Calculate metrics
    test_mse = mean_squared_error(test_true, test_preds)
    test_mae = mean_absolute_error(test_true, test_preds)
    
    print(f'Test Results:')
    print(f'MSE: {test_mse:.4f}')
    print(f'MAE: {test_mae:.4f}')
    print(f'Test Preds: {test_preds}')
    print(f'Test True: {test_true}')
    
    return test_mse, test_mae



# Example usage:
if __name__ == "__main__":
    # Initialize data loaders
    train_loader, val_loader, test_loader = prepare_data()
    # Initialize model
    model = DualXRayNet(num_classes=1, pretrained=True, freeze_backbone=True)
    
    #######################################
    # Load the pretrained model checkpoint
    checkpoint_path = "/workspace/swapnil/xray/checkpoints/full_frozen_train_loss_pat_3/best_model.pth"  # Replace with your model's checkpoint path
    model.load_state_dict(torch.load(checkpoint_path))

#     # Unfreeze all layers
#     for param in model.parameters():
#         param.requires_grad = True

    # Unfreeze only the last few layers of the backbone
    for name, param in model.front_backbone.named_parameters():
        if "layer4" in name or "fc" in name:  # Unfreeze layer4 and fc in the front backbone
            param.requires_grad = True
    
    for name, param in model.side_backbone.named_parameters():
        if "layer4" in name or "fc" in name:  # Unfreeze layer4 and fc in the side backbone
            param.requires_grad = True

#     # Verify trainable parameters
#     print("Trainable parameters in front backbone:")
#     for name, param in model.front_backbone.named_parameters():
#         print(name, param.requires_grad)

#     print("Trainable parameters in side backbone:")
#     for name, param in model.side_backbone.named_parameters():
#         print(name, param.requires_grad)
    
#     print("Trainable parameters in classifier:")
#     for name, param in model.classifier.named_parameters():
#         print(name, param.requires_grad)
    ######################################
    
    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
