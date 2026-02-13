import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.model import NCF
from src.dataset import get_dataloader, preprocess_data
from src.utils import load_config, save_checkpoint, get_device

def train():
    # Load Configuration
    config = load_config()
    
    # Setup Device
    device = get_device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Load and Preprocess Data (Mock data if file doesn't exist)
    data_path = config['data']['raw_path']
    if not os.path.exists(data_path):
        print("Generating mock data for demonstration...")
        # Create mock data: 1000 users, 500 items, 10000 interactions
        num_users, num_items = 1000, 500
        mock_data = {
            'user_id': torch.randint(0, num_users, (10000,)).tolist(),
            'item_id': torch.randint(0, num_items, (10000,)).tolist(),
            'label': torch.randint(0, 2, (10000,)).tolist()
        }
        df = pd.DataFrame(mock_data)
    else:
        df = pd.read_csv(data_path)
    
    df, user_map, item_map = preprocess_data(df)
    num_users = len(user_map)
    num_items = len(item_map)
    
    # Split Data
    train_df, val_df = train_test_split(df, test_size=config['data']['test_size'], random_state=42)
    
    # Create DataLoaders
    train_loader = get_dataloader(train_df, config['training']['batch_size'])
    val_loader = get_dataloader(val_df, config['training']['batch_size'], shuffle=False)
    
    # Initialize Model
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config['model']['embedding_dim'],
        layers=config['model']['layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay']
    )
    
    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        for users, items, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                outputs = model(users, items)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Early Stopping & Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, config['inference']['model_path'])
            print("Checkpoint saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['training']['early_stopping_patience']:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train()
