import torch
import torch.nn.functional as F
import math

from src.embeddings import get_sinusoidal_positional_embeddings, TokenEmbedding
from src.block import TransformerBlock
from src.masks import make_causal_mask, make_padding_mask, combine_masks
from src.word_tokenizer import decode_tokens_words, encode_text_words


class SpellforgeTransformer(torch.nn.Module):
    """
    Complete transformer language model for D&D spell generation
    
    Architecture:
    1. Token embeddings + positional encodings
    2. Stack of transformer blocks (attention + FFN)
    3. Language modeling head (linear projection to vocab)
    
    Input: [batch_size, seq_len] token IDs
    Output: [batch_size, seq_len, vocab_size] logits for next token prediction
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=None,
        max_seq_len=1024,
        dropout=0.1,
        activation="gelu",
        bias=True,
        pad_token_id=0
    ):
        super().__init__()
        
        # Input validation
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # token embeddings: convert token IDs to dense vectors using your TokenEmbedding class
        # each token ID gets mapped to a d_model dimensional vector
        self.token_embeddings = TokenEmbedding(vocab_size, d_model)
        
        # dropout for embeddings (helps prevent overfitting)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        
        # stack of transformer blocks
        # each block has self-attention + feed-forward + residual connections
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                activation=activation,
                dropout=dropout,
                bias=bias
            ) for _ in range(num_layers)
        ])
        
        # final layer normalization (helps with training stability)
        self.final_norm = torch.nn.LayerNorm(d_model)
        
        # language modeling head: project to vocabulary size for next token prediction
        # gives logits for every possible token in vocabulary
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        
        # initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for stable training"""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                # xavier/glorot initialization for linear layers
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, TokenEmbedding):
                # normal initialization for your token embedding
                torch.nn.init.normal_(module.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, return_loss=False, labels=None):
        """
        Forward pass through the transformer
        
        Args:
            input_ids: [batch_size, seq_len] - token IDs
            attention_mask: [batch_size, seq_len] - mask for padding (1=real, 0=pad)
            return_loss: whether to compute and return loss
            labels: [batch_size, seq_len] - target tokens for loss computation
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - probability distribution over vocab
            loss: scalar tensor (if return_loss=True and labels provided)
        """
        if not torch.is_tensor(input_ids):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, seq_len]")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # check sequence length doesn't exceed maximum
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # step 1: convert token IDs to embeddings
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        token_embeds = self.token_embeddings(input_ids)
        
        # step 2: add positional encodings so model knows token positions
        # [1, seq_len, d_model] - broadcasts to batch dimension
        pos_embeds = get_sinusoidal_positional_embeddings(seq_len, self.d_model, device)
        
        # combine token and position information
        # [batch_size, seq_len, d_model] + [1, seq_len, d_model] = [batch_size, seq_len, d_model]
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)
        
        # step 3: create attention masks for the transformer
        # causal mask: each token can only attend to previous tokens (for generation)
        causal_mask = make_causal_mask(seq_len, device)  # [1, 1, seq_len, seq_len]
        
        # padding mask: ignore padded tokens in attention
        if attention_mask is not None:
            # convert attention_mask to proper format for attention
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # expand to match sequence dimensions for combine_masks
            padding_mask = padding_mask.expand(-1, -1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
        else:
            # create padding mask from input_ids if not provided
            padding_mask = make_padding_mask(input_ids, self.pad_token_id)  # [batch_size, 1, 1, seq_len]
            # expand to match sequence dimensions for combine_masks
            padding_mask = padding_mask.expand(-1, -1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
        
        # combine causal and padding masks using your combine_masks function
        # causal_mask will broadcast from [1, 1, seq_len, seq_len] to [batch_size, 1, seq_len, seq_len]
        combined_mask = combine_masks(causal_mask, padding_mask)
        
        # step 4: pass through transformer blocks
        # each block applies self-attention and feed-forward with residual connections
        for block in self.transformer_blocks:
            x = block(x, attn_mask=combined_mask)  # [batch_size, seq_len, d_model]
        
        # step 5: final layer normalization
        x = self.final_norm(x)  # [batch_size, seq_len, d_model]
        
        # step 6: project to vocabulary space for next token prediction
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # step 7: compute loss if training
        loss = None
        if return_loss and labels is not None:
            # shift logits and labels for next-token prediction
            # predict token at position i+1 from tokens 0 to i
            shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
            shift_labels = labels[..., 1:].contiguous()      # [batch, seq_len-1]
            
            # flatten for cross-entropy computation
            shift_logits = shift_logits.view(-1, self.vocab_size)  # [batch*(seq_len-1), vocab]
            shift_labels = shift_labels.view(-1)                   # [batch*(seq_len-1)]
            
            # cross-entropy loss (ignores padding tokens automatically)
            loss = F.cross_entropy(
                shift_logits, 
                shift_labels, 
                ignore_index=self.pad_token_id,
                reduction='mean'
            )
        
        if return_loss:
            return logits, loss
        return logits
    
    def forward_lm_batch(self, input_ids, target_ids, eos_weight=1.0):
        """
        Forward pass for language model batches with EOS token weighting
        
        PROBLEM AFTER EOS WEIGHTING:
        ============================
        ISSUE: Even with 20x EOS weight, model still never generates EOS tokens during inference
        
        ROOT CAUSE DISCOVERED: Data Format Mismatch
        - Training data had: "Well >>>\n\n\n<EOS>" (EOS after newlines)
        - Generation expected: "Well >>><EOS>" (EOS immediately after)
        - Model correctly learned the training pattern, but it didn't match generation expectation
        
        SOLUTION: Fixed Data Format in preprocessing.py
        - Removed extra newlines between spells
        - Now training data has: "Well >>><EOS>" (immediate EOS)
        - This matches generation expectation exactly
        - Simple EOS weighting should now be sufficient
        
        Args:
            input_ids: [batch_size, seq_len-1] - input tokens
            target_ids: [batch_size, seq_len-1] - target tokens (shifted)
            eos_weight: float - weight multiplier for EOS tokens (default 1.0 = no weighting)
        
        Returns:
            loss: scalar tensor - cross-entropy loss
            logits: [batch_size, seq_len-1, vocab_size] - predictions
        """
        # get logits for input sequence
        logits = self(input_ids)  # [batch_size, seq_len-1, vocab_size]
        
        # flatten for cross-entropy loss computation
        flat_logits = logits.view(-1, self.vocab_size)  # [batch_size*(seq_len-1), vocab_size]
        flat_targets = target_ids.view(-1)  # [batch_size*(seq_len-1)]
        
        if eos_weight > 1.0:
            # Use weighted loss to emphasize EOS tokens
            # Create weight tensor - most tokens get weight 1.0, EOS gets higher weight
            weights = torch.ones_like(flat_targets, dtype=torch.float)
            weights[flat_targets == 2] = eos_weight  # EOS token (ID=2) gets higher weight
            weights[flat_targets == 0] = 0.0  # PAD tokens get zero weight (ignored)
            
            # Compute loss manually with weights
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(flat_logits, flat_targets)
            loss = (losses * weights).sum() / weights.sum()
        else:
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.pad_token_id,
                reduction='mean'
            )
        
        return loss, logits
    
    def generate(
        self, 
        prompt_ids, 
        max_new_tokens=100, 
        temperature=1.0, 
        top_k=None, 
        top_p=None,
        do_sample=True,
        eos_token_id=2,  # <EOS> token from your tokenizer
        bos_token_id=1,  # <BOS> token from your tokenizer
        pad_token_id=0   # <PAD> token from your tokenizer
    ):
        """
        Generate new spell text autoregressively
        
        Args:
            prompt_ids: [1, prompt_len] or [prompt_len] - starting tokens
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: keep only top k tokens for sampling
            top_p: nucleus sampling - keep tokens with cumulative prob <= top_p
            do_sample: if False, use greedy decoding
            eos_token_id: stop generation when this token is generated
            pad_token_id: padding token ID
        
        Returns:
            generated_ids: [1, total_len] - prompt + generated tokens
        """
        self.eval()  # set to evaluation mode
        
        # ensure prompt_ids is 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)  # [1, prompt_len]
        
        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        # if prompt is empty or just BOS token, start with BOS
        if prompt_len == 0 or (prompt_len == 1 and prompt_ids[0, 0].item() == bos_token_id):
            generated_ids = torch.tensor([[bos_token_id]], device=device, dtype=torch.long)
        else:
            # start with the provided prompt
            generated_ids = prompt_ids.clone()  # [1, prompt_len]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # get current sequence length
                current_len = generated_ids.shape[1]
                
                # stop if we exceed max sequence length
                if current_len >= self.max_seq_len:
                    break
                
                # forward pass to get next token logits
                logits = self(generated_ids)  # [1, current_len, vocab_size]
                
                # get logits for the last position (next token prediction)
                next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                
                # EOS PATTERN DETECTION: Temporarily disabled for word-level tokenization testing
                # if current_len >= 50:  # Only after generating substantial content
                #     # Get recent tokens to build text context
                #     recent_tokens = generated_ids[0, -15:].tolist() if current_len >= 15 else generated_ids[0, :].tolist()
                #     
                #     # Convert recent tokens to text for pattern matching
                #     recent_chars = []
                #     for token_id in recent_tokens:
                #         if token_id < self.vocab_size:  # Valid token ID
                #             # Use ASCII approximation for quick pattern detection
                #             if 32 <= token_id <= 126:  # Printable ASCII range
                #                 recent_chars.append(chr(token_id))
                #     
                #     recent_text = ''.join(recent_chars[-10:])  # Last ~10 characters
                #     
                #     # Check for spell ending patterns
                #     ending_indicators = ['May it', 'Serve', 'You', 'Well', '>>']
                #     pattern_found = any(indicator in recent_text for indicator in ending_indicators)
                #     
                #     # Also check for the >>> token pattern
                #     if current_len >= 3:
                #         last_three = generated_ids[0, -3:].tolist()
                #         if last_three == [17, 17, 17]:  # >>> pattern
                #             pattern_found = True
                #     
                #     if pattern_found:
                #         # Boost EOS token probability
                #         next_token_logits[0, eos_token_id] += 15.0
                #         print(f"Spell ending pattern detected! Boosting EOS...")
                
                # Apply temperature scaling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # apply top-k filtering
                if top_k is not None:
                    # keep only top k tokens
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # set all other logits to -inf
                    next_token_logits.fill_(-float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # apply nucleus (top-p) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # shift right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # sample next token
                if do_sample:
                    # multinomial sampling
                    probs = F.softmax(next_token_logits, dim=-1)  # [1, vocab_size]
                    next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                else:
                    # greedy decoding (choose most likely token)
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]
                
                # append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)  # [1, current_len+1]
                
                # stop if EOS token is generated
                if next_token.item() == eos_token_id:
                    break
        
        return generated_ids
    
    def prepare_generation_prompt(self, text_prompt="", token_to_id=None, bos_token_id=1):
        """
        Helper method to prepare prompts for generation from text
        
        Args:
            text_prompt: string to use as prompt (e.g., "<<< New Spell Forged >>>")
            token_to_id: token-to-ID mapping from the tokenizer
            bos_token_id: BOS token ID
        
        Returns:
            prompt_ids: [1, prompt_len] tensor ready for generation
        """
        if token_to_id is None:
            # return just BOS token if no vocab provided
            return torch.tensor([[bos_token_id]], dtype=torch.long)

        prompt_tokens = encode_text_words(
            text_prompt,
            token_to_id,
            add_bos=True,
            add_eos=False,
        )
        return torch.tensor([prompt_tokens], dtype=torch.long)
    
    def get_num_params(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizers(self, learning_rate=1e-4, weight_decay=0.01):
        """
        Configure AdamW optimizer with weight decay
        Excludes embeddings and biases from weight decay
        """
        # separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                # don't apply weight decay to embeddings, biases, or layer norms
                if 'bias' in name or 'norm' in name or 'token_embeddings' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)
        return optimizer
    
    def decode_generation(self, generated_ids, id_to_token=None, skip_special_tokens=True):
        """
        Helper method to decode generated token IDs back to text
        
        Args:
            generated_ids: [1, seq_len] tensor from generation
            id_to_token: ID-to-token mapping from the tokenizer
            skip_special_tokens: whether to skip <BOS>, <EOS>, <PAD> tokens
        
        Returns:
            decoded_text: string representation of generated tokens
        """
        if id_to_token is None:
            return f"Token IDs: {generated_ids.squeeze(0).tolist()}"

        token_ids = generated_ids.squeeze(0).tolist()
        if not skip_special_tokens:
            return " ".join(id_to_token.get(token_id, f"<UNK:{token_id}>") for token_id in token_ids)

        return decode_tokens_words(token_ids, id_to_token)
