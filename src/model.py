import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) Model.
    
    Architecture:
    - User Embedding Layer
    - Item Embedding Layer
    - Concatenation of Embeddings
    - Multi-Layer Perceptron (MLP)
    - Output Layer with Sigmoid Activation
    """
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        
        # User and Item Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP Layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            input_size = layer_size
        
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Output layer
        self.prediction = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.prediction.weight)

    def forward(self, user_indices, item_indices):
        """
        Forward pass of the model.
        
        Args:
            user_indices (torch.Tensor): Batch of user IDs.
            item_indices (torch.Tensor): Batch of item IDs.
            
        Returns:
            torch.Tensor: Predicted interaction probability (0 to 1).
        """
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Concatenate embeddings
        vector = torch.cat([user_embed, item_embed], dim=-1)
        
        # Pass through MLP
        mlp_output = self.mlp(vector)
        
        # Prediction
        prediction = self.prediction(mlp_output)
        return self.sigmoid(prediction).view(-1)
