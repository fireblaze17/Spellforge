import torch
import math
from tqdm import tqdm


def simple_train(
    model, 
    train_batches, 
    val_batches, 
    charToId, 
    idToChar,
    epochs=15,
    learning_rate=3e-4,
    device='auto'
):
    """
    Simple training loop - just the essentials!
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    print(f"🚀 Training on {device}")
    
    # Simple optimizer
    optimizer = model.configure_optimizers(learning_rate=learning_rate)
    
    pad_id = charToId.get('<PAD>', 0)
    bos_id = charToId.get('<BOS>', 1)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_batches, desc=f"Epoch {epoch+1}")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            loss, _ = model.forward_lm_batch(input_ids, target_ids)
            loss.backward()
            
            # Simple gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        avg_train_loss = total_loss / len(train_batches)
        
        # Simple validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_batches:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                loss, _ = model.forward_lm_batch(input_ids, target_ids)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_batches)
        perplexity = math.exp(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.3f}, Val Loss={avg_val_loss:.3f}, Perplexity={perplexity:.1f}")
        
        # Generate sample every few epochs
        if epoch % 3 == 0:
            sample = generate_simple_sample(model, charToId, idToChar, device)
            print(f"Sample: {sample[:100]}...")
        
        # Simple checkpoint save
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
    
    print("✅ Training complete!")


def generate_simple_sample(model, charToId, idToChar, device):
    """Generate a simple spell sample"""
    model.eval()
    
    # Start with BOS token
    prompt_ids = torch.tensor([[charToId.get('<BOS>', 1)]], device=device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt_ids, 
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True
        )
    
    # Decode to text
    tokens = generated.squeeze().tolist()
    text = ""
    for token_id in tokens:
        if token_id < len(idToChar):
            char = idToChar[token_id]
            if char not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                text += char
    
    return text