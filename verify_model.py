import torch
from llm.model import GPT, GPTConfig

def verify_gpt():
    config = GPTConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    print("Model instantiated successfully.")
    
    # Create dummy input
    x = torch.randint(0, 100, (4, 32)) # (B, T)
    
    # Forward pass
    logits, loss = model(x)
    print(f"Logits shape: {logits.shape}")
    
    if logits.shape == (4, 32, 100):
        print("Logits shape is correct.")
    else:
        print(f"Logits shape mismatch: expected (4, 32, 100), got {logits.shape}")

    # Forward pass with targets
    targets = torch.randint(0, 100, (4, 32))
    logits, loss = model(x, targets)
    print(f"Loss: {loss.item()}")
    
    if loss is not None:
         print("Loss calculation works.")
    else:
         print("Loss calculation failed.")

if __name__ == "__main__":
    verify_gpt()
