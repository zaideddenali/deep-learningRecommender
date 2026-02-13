import torch
from src.model import NCF
from src.utils import load_config, get_device, load_checkpoint
import os

class RecommenderInference:
    def __init__(self, num_users, num_items):
        self.config = load_config()
        self.device = get_device(self.config['training']['device'])
        
        self.model = NCF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.config['model']['embedding_dim'],
            layers=self.config['model']['layers']
        ).to(self.device)
        
        if os.path.exists(self.config['inference']['model_path']):
            self.model = load_checkpoint(self.model, self.config['inference']['model_path'], self.device)
            self.model.eval()
        else:
            print("Warning: Model checkpoint not found. Using uninitialized model.")

    def recommend(self, user_id, all_item_ids, top_k=10):
        """Generate top-K recommendations for a user."""
        user_tensor = torch.full((len(all_item_ids),), user_id, dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor)
        
        # Get top-K indices
        top_k_scores, top_k_indices = torch.topk(scores, min(top_k, len(scores)))
        
        recommended_items = [all_item_ids[i] for i in top_k_indices.cpu().numpy()]
        return list(zip(recommended_items, top_k_scores.cpu().numpy()))

if __name__ == "__main__":
    # Example usage
    # Note: In a real scenario, you'd load the mapping from training
    inference = RecommenderInference(num_users=1000, num_items=500)
    items_to_rank = list(range(100))
    recs = inference.recommend(user_id=1, all_item_ids=items_to_rank, top_k=5)
    print(f"Top 5 Recommendations for User 1: {recs}")
