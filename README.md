# LLM From Scratch

This project is an educational implementation of a Large Language Model (LLM) built from scratch using PyTorch. The goal is to understand the inner workings of Transformer-based models, similar to Andrej Karpathy's nanoGPT.

## Project Structure

Here is a breakdown of the project components:

- **`.gitignore`**: Specifies files and directories that Git should ignore, such as virtual environments (`.venv`), cache files (`__pycache__`), and sensitive data.
- **`.python-version`**: Defines the Python version used in this project (managed by `uv`).
- **`pyproject.toml`**: The main configuration file for the project. It lists dependencies, project metadata, and build settings. `uv` uses this file to manage the environment.
- **`README.md`**: The documentation file you are currently reading.
- **`config/`**: Contains configuration files for training runs (e.g., hyperparameters, model size, learning rate).
  - `train_gpt2.py`: Example configuration for training a GPT-style model.
- **`data/`**: Stores raw and processed datasets.
  - `prepare.py`: Scripts to download and tokenize data (e.g., TinyShakespeare, OpenWebText) before training.
- **`llm/`**: The core source code for the language model.
  - `model.py`: Defines the Transformer architecture (Attention, MLP, LayerNorm, etc.).
  - `train.py`: Implements the training loop, including forward pass, backward pass, and optimization.
  - `tokenizer.py`: Handles text-to-token encoding and decoding.
- **`notebooks/`**: Jupyter notebooks for interactive experimentation and visualization.
  - `exploration.ipynb`: A playground for testing model components and data loading.

## Getting Started

1.  **Install `uv`**:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Dependencies**:

    ```bash
    uv sync
    ```

3.  **Prepare Data**:

    ```bash
    uv run data/prepare.py
    ```

4.  **Train the Model**:
    ```bash
    uv run llm/train.py config/train_gpt2.py
    ```
