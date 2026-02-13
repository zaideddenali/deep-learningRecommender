# 🚀 Deep Learning Recommendation System at Scale

## Production-Ready Deep Learning Recommendation System

Modern platforms like Netflix, Amazon, Spotify, and YouTube heavily rely on sophisticated deep learning recommendation systems trained on massive user interaction data. This project aims to build a **scalable deep learning recommendation system** that mirrors a real-world production environment, significantly strengthening your profile in Machine Learning Engineering.

This is not a toy recommender; it is structured like a **production-ready deep learning ML system**.

---

## 📌 Project Overview

This project implements a deep learning recommendation system using:

*   **Embedding layers**: To learn dense representations of users and items.
*   **Neural Collaborative Filtering (NCF)**: To capture non-linear user-item interactions.
*   **PyTorch**: As the primary deep learning framework.
*   **Batch training pipeline**: For efficient processing of large datasets.
*   **Modular ML engineering structure**: Promoting maintainability and scalability.
*   **Model evaluation & explainability**: Focusing on relevant ranking metrics.
*   **Clean documentation layer**: Including a detailed code explanation.

---

## 🎯 Problem Statement

To build a deep learning recommendation system that:

*   Learns effective user and item embeddings.
*   Predicts interaction probability between users and items.
*   Scales efficiently to large datasets.
*   Can serve real-time recommendations.
*   Is modular, maintainable, and production-ready.

---

## 🧠 Model Approach: Neural Collaborative Filtering (NCF)

The core model implemented is **Neural Collaborative Filtering (NCF)**. This architecture is designed to model the interaction function between users and items using neural networks.

### Architecture:

1.  **User ID & Item ID Embeddings**: Each user and item is mapped to a dense vector representation.
2.  **Concatenation**: The user and item embeddings are concatenated to form a single interaction vector.
3.  **Dense Layers (MLP)**: The concatenated vector is passed through multiple fully connected layers with non-linear activation functions (ReLU) and Dropout for regularization.
4.  **Output Layer**: A final linear layer with a Sigmoid activation predicts the interaction probability (e.g., likelihood of a user clicking on an item).

This approach effectively captures complex, non-linear relationships between users and items that traditional matrix factorization methods might miss.

### Example Architecture Flow:

```
UserEmbedding(num_users, embedding_dim)
ItemEmbedding(num_items, embedding_dim)

→ Concatenate
→ Linear(input_dim → layer_1_size) → ReLU → Dropout
→ Linear(layer_1_size → layer_2_size) → ReLU
→ Linear(layer_2_size → 1)
→ Sigmoid
```

---

## 📂 Project Structure

```
deep-learning-recommender/
│
├── data/                 # Raw and processed datasets
│   ├── raw/              # Original datasets (e.g., ratings.csv)
│   └── processed/        # Intermediate processed data
│
├── src/                  # Core source code
│   ├── __init__.py       # Python package initializer
│   ├── dataset.py        # Custom PyTorch Dataset and DataLoader
│   ├── model.py          # NCF model definition
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│   ├── inference.py      # Recommendation inference script
│   └── utils.py          # Utility functions (config, checkpointing, device)
│
├── configs/              # Configuration files
│   └── config.yaml       # Hyperparameters and settings
│
├── notebooks/            # Jupyter notebooks for exploration
│   └── exploration.ipynb # Example notebook
│
├── docs/                 # Project documentation
│   └── CODE_EXPLANATION.md # Detailed explanation of the codebase
│
├── tests/                # Unit and integration tests
│   └── test_model.py     # Example tests
│
├── requirements.txt      # Python dependencies
├── README.md             # Project overview and setup
└── .gitignore            # Git ignore file
```

---

## 🔥 Core Components Explained

*   **`dataset.py`**: Defines `RecommenderDataset` for handling user-item pairs and `get_dataloader` for efficient batching. Includes `preprocess_data` for ID mapping.
*   **`model.py`**: Contains the `NCF` class, implementing the neural network architecture with embedding layers, fully connected layers, dropout, and a sigmoid output.
*   **`train.py`**: Manages the training loop, including loss function (`BCELoss`), optimizer (`Adam`), early stopping, and model checkpoint saving. It can generate mock data if `data/raw/ratings.csv` is not present.
*   **`evaluate.py`**: Calculates key ranking metrics such as AUC. It loads a trained model and evaluates its performance on a test set.
*   **`inference.py`**: Provides functionality to load a trained model and generate top-N recommendations for given users.
*   **`utils.py`**: Houses helper functions for loading `config.yaml`, saving/loading model checkpoints, and determining the appropriate computing device (CPU/GPU).
*   **`config.yaml`**: Centralized configuration for model hyperparameters, training settings, and data paths.

---

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/deep-learning-recommender.git
    cd deep-learning-recommender
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    Place your user-item interaction data (e.g., `ratings.csv` with `user_id`, `item_id`, `label` columns) in the `data/raw/` directory. If no data is provided, `train.py` will generate mock data for demonstration purposes.

---

## 🚀 Usage

### Training the Model

To train the NCF model, run the `train.py` script:

```bash
python src/train.py
```

The trained model checkpoints will be saved in the `checkpoints/` directory as specified in `config.yaml`.

### Evaluating the Model

To evaluate the performance of the trained model, run the `evaluate.py` script:

```bash
python src/evaluate.py
```

This will output metrics like AUC based on the test set.

### Generating Recommendations

To use the trained model for inference and generate recommendations, you can run `inference.py`:

```bash
python src/inference.py
```

Modify the `if __name__ == "__main__":` block in `inference.py` to test with specific user and item IDs.

---

## 📄 Documentation

*   **`docs/CODE_EXPLANATION.md`**: Provides an in-depth explanation of the system architecture, model details, data flow, training logic, and engineering decisions. This is a crucial document for understanding the project's internals.

---

## 🧪 Testing

An example test file `tests/test_model.py` is included. You can run tests using `pytest`:

```bash
pytest tests/
```

---

## 🌟 What This Project Proves

This project demonstrates advanced skills in:

*   Implementing a deep learning recommender in PyTorch from scratch.
*   Designing and utilizing embedding architectures.
*   Developing custom PyTorch `Dataset` and training loops.
*   Evaluating recommendation systems with ranking metrics.
*   Structuring an ML project like a production system.
*   Separating training and inference pipelines.
*   Creating comprehensive engineering documentation.


