import torch
import unittest
from src.model import NCF

class TestNCFModel(unittest.TestCase):

    def setUp(self):
        self.num_users = 100
        self.num_items = 50
        self.embedding_dim = 16
        self.layers = [32, 16]
        self.dropout = 0.1
        self.model = NCF(self.num_users, self.num_items, self.embedding_dim, self.layers, self.dropout)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, NCF)
        self.assertIsInstance(self.model.user_embedding, torch.nn.Embedding)
        self.assertIsInstance(self.model.item_embedding, torch.nn.Embedding)
        self.assertIsInstance(self.model.mlp, torch.nn.Sequential)
        self.assertIsInstance(self.model.prediction, torch.nn.Linear)

    def test_forward_pass_output_shape(self):
        batch_size = 4
        user_indices = torch.randint(0, self.num_users, (batch_size,))
        item_indices = torch.randint(0, self.num_items, (batch_size,))
        
        output = self.model(user_indices, item_indices)
        self.assertEqual(output.shape, (batch_size,))

    def test_forward_pass_output_range(self):
        batch_size = 4
        user_indices = torch.randint(0, self.num_users, (batch_size,))
        item_indices = torch.randint(0, self.num_items, (batch_size,))
        
        output = self.model(user_indices, item_indices)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_embedding_dimensions(self):
        self.assertEqual(self.model.user_embedding.embedding_dim, self.embedding_dim)
        self.assertEqual(self.model.item_embedding.embedding_dim, self.embedding_dim)

if __name__ == '__main__':
    unittest.main()
