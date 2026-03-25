from pathlib import Path

import torch
import math
from tqdm import tqdm

from src.word_tokenizer import decode_tokens_words


def simple_train(
    model, 
    train_batches, 
    val_batches, 
    token_to_id,
    id_to_token,
    epochs=30,
    learning_rate=1.5e-4,
    device='auto',
    eos_weight=25.0,
    output_dir="artifacts/models",
    sample_every_n_epochs=4,
    checkpoint_every_n_epochs=5,
):
    """
    Train the recipe language model and save checkpoints to a dedicated artifact directory.
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    print(f"Training on {device}")
    
    # Optimizer with proven settings
    optimizer = model.configure_optimizers(learning_rate=learning_rate, weight_decay=0.01)
    
    # Add learning rate scheduling for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.1)
    
    pad_id = token_to_id.get('<PAD>', 0)
    eos_id = token_to_id.get('<EOS>', 2)
    
    best_val_loss = float('inf')
    best_model_path = output_dir / "best_model.pt"
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        eos_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(train_batches, desc=f"Epoch {epoch+1:2d}")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            loss, logits = model.forward_lm_batch(input_ids, target_ids, eos_weight=eos_weight)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Track EOS predictions during training
            predictions = torch.argmax(logits, dim=-1).flatten()
            targets = target_ids.flatten()
            valid_targets = targets != pad_id
            
            eos_predictions += (predictions[valid_targets] == eos_id).sum().item()
            total_predictions += valid_targets.sum().item()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            eos_rate = eos_predictions / max(total_predictions, 1) * 100
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'lr': f'{current_lr:.1e}',
                'eos%': f'{eos_rate:.1f}'
            })
        
        scheduler.step()  # Update learning rate
        
        avg_train_loss = total_loss / len(train_batches)
        eos_rate = eos_predictions / max(total_predictions, 1) * 100
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, target_ids in val_batches:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                loss, _ = model.forward_lm_batch(input_ids, target_ids, eos_weight=eos_weight)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_batches)
        perplexity = math.exp(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d}: Train={avg_train_loss:.3f}, Val={avg_val_loss:.3f}, PPL={perplexity:.1f}, EOS={eos_rate:.1f}%, LR={scheduler.get_last_lr()[0]:.1e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {avg_val_loss:.3f} validation loss")
        
        # Generate sample every few epochs
        if epoch % sample_every_n_epochs == 0 or epoch == epochs - 1:
            sample = generate_simple_sample(model, token_to_id, id_to_token, device)
            print(f"Sample recipe:\n{sample}\n{'-'*60}")
        
        # Checkpoint saves
        if epoch % checkpoint_every_n_epochs == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Save final model
    final_model_path = output_dir / f"final_model_epoch_{epochs}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")
    print(f"Training complete! Best validation loss: {best_val_loss:.3f}")
    return {
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "best_val_loss": best_val_loss,
    }


def generate_simple_sample(model, token_to_id, id_to_token, device):
    """Generate a sample with better parameters."""
    model.eval()
    
    # Start with BOS token
    prompt_ids = torch.tensor([[token_to_id.get('<BOS>', 1)]], device=device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt_ids, 
            max_new_tokens=600,  # More tokens for complete recipes
            temperature=0.8,     # Balanced creativity
            top_k=40,            # Keep generation focused on likely next tokens
            repetition_penalty=1.15,  # Light penalty to reduce loops and duplicates
            do_sample=True,
            eos_token_id=token_to_id.get('<EOS>', 2)
        )
    
    return decode_tokens_words(generated.squeeze().tolist(), id_to_token)
