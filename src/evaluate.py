import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils import load_config, get_device, load_checkpoint
from src.model import NCF
from src.dataset import get_dataloader, preprocess_data
import pandas as pd
import os

def calculate_metrics(model, dataloader, device):
    """Calculate AUC and other metrics."""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for users, items, labels in dataloader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            outputs = model(users, items)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
            
    auc = roc_auc_score(all_labels, all_preds)
    return {"AUC": auc}

def evaluate():
    config = load_config()
    device = get_device(config['training']['device'])
    
    # Load data (using same logic as train.py for demo)
    data_path = config['data']['raw_path']
    if not os.path.exists(data_path):
        # This is just for demonstration if raw data is missing
        num_users, num_items = 1000, 500
        mock_data = {
            'user_id': torch.randint(0, num_users, (2000,)).tolist(),
            'item_id': torch.randint(0, num_items, (2000,)).tolist(),
            'label': torch.randint(0, 2, (2000,)).tolist()
        }
        df = pd.DataFrame(mock_data)
    else:
        df = pd.read_csv(data_path)

    df, user_map, item_map = preprocess_data(df)
    test_loader = get_dataloader(df, config['training']['batch_size'], shuffle=False)
    
    # Load Model
    model = NCF(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=config['model']['embedding_dim'],
        layers=config['model']['layers']
    ).to(device)
    
    if os.path.exists(config['inference']['model_path']):
        model = load_checkpoint(model, config['inference']['model_path'], device)
        metrics = calculate_metrics(model, test_loader, device)
        print(f"Evaluation Metrics: {metrics}")
    else:
        print("Model checkpoint not found. Please train the model first.")

if __name__ == "__main__":
    evaluate()
