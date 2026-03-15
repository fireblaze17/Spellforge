import torch
import torch.nn.functional as F
import math
import re
from typing import Dict, List, Tuple, Optional


class SpellEvaluator:
    """
    Comprehensive evaluation system for the Spellforge transformer.
    Tests both quantitative metrics and qualitative spell generation.
    """
    
    def __init__(self, model, charToId, idToChar, device='cpu'):
        self.model = model
        self.charToId = charToId
        self.idToChar = idToChar
        self.device = device
        
        # Special token IDs
        self.pad_id = charToId.get('<PAD>', 0)
        self.bos_id = charToId.get('<BOS>', 1)
        self.eos_id = charToId.get('<EOS>', 2)
        self.unk_id = charToId.get('<UNK>', 3)
    
    def compute_perplexity(self, val_batches) -> float:
        """
        Compute perplexity on validation batches.
        Lower perplexity = better model.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_batches:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                loss, _ = self.model.forward_lm_batch(input_ids, target_ids)
                
                # Count non-padding tokens for accurate perplexity
                non_pad_tokens = (target_ids != self.pad_id).sum().item()
                
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
    
    def compute_token_accuracy(self, val_batches) -> float:
        """
        Compute next-token prediction accuracy on validation set.
        """
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_batches:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                _, logits = self.model.forward_lm_batch(input_ids, target_ids)
                predictions = torch.argmax(logits, dim=-1)
                
                # Only count non-padding tokens
                mask = (target_ids != self.pad_id)
                correct = (predictions == target_ids) & mask
                
                correct_predictions += correct.sum().item()
                total_predictions += mask.sum().item()
        
        if total_predictions == 0:
            return 0.0
        
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def encode_prompt(self, text_prompt: str) -> torch.Tensor:
        """Helper to encode text prompts to token IDs"""
        prompt_tokens = [self.bos_id]
        for char in text_prompt:
            if char in self.charToId:
                prompt_tokens.append(self.charToId[char])
            else:
                prompt_tokens.append(self.unk_id)
        
        return torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    
    def decode_tokens(self, token_ids: torch.Tensor, skip_special=True) -> str:
        """Helper to decode token IDs back to text"""
        if token_ids.ndim > 1:
            token_ids = token_ids.squeeze()
        
        decoded_chars = []
        special_tokens = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        
        for token_id in token_ids.tolist():
            if skip_special and token_id in special_tokens:
                continue
            
            if token_id < len(self.idToChar):
                decoded_chars.append(self.idToChar[token_id])
        
        return ''.join(decoded_chars)
    
    def generate_sample_spells(self, num_samples=5, temperature=0.8) -> List[str]:
        """
        Generate sample spells with different prompts to test creativity.
        """
        sample_prompts = [
            "<<< New Spell Forged >>>\n  Name: ",
            "<<< New Spell Forged >>>\n  Name: Fireball",
            "<<< New Spell Forged >>>\n  Name: Healing Light\n  Classes: ",
            "<<< New Spell Forged >>>\n  Name: Shadow Strike\n  Classes: Rogue\n  School: ",
            "<<< New Spell Forged >>>\n  Name: Magic Missile\n  Classes: Wizard\n  School: Evocation\n  Range: "
        ]
        
        generated_spells = []
        self.model.eval()
        
        for i, prompt in enumerate(sample_prompts[:num_samples]):
            prompt_ids = self.encode_prompt(prompt)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=300,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.eos_id
                )
            
            generated_text = self.decode_tokens(generated_ids)
            generated_spells.append(f"Sample {i+1}:\n{generated_text}\n")
        
        return generated_spells
    
    def check_format_compliance(self, generated_text: str) -> Dict[str, bool]:
        """
        Check if generated spell follows expected D&D format.
        """
        checks = {
            'starts_with_forge': generated_text.strip().startswith('<<< New Spell Forged >>>'),
            'ends_with_serve': '<<< May it Serve You Well >>>' in generated_text,
            'has_name_field': 'Name:' in generated_text,
            'has_classes_field': 'Classes:' in generated_text,
            'has_school_field': 'School:' in generated_text,
            'has_range_field': 'Range:' in generated_text,
            'has_duration_field': 'Duration:' in generated_text,
            'has_description_field': 'Description:' in generated_text,
        }
        
        return checks
    
    def evaluate_model(self, val_batches, num_samples=5, temperature=0.8) -> Dict:
        """
        Complete model evaluation with metrics and sample generation.
        """
        print("🔍 Starting comprehensive model evaluation...")
        
        # Quantitative metrics
        print("Computing perplexity...")
        perplexity = self.compute_perplexity(val_batches)
        
        print("Computing token accuracy...")
        accuracy = self.compute_token_accuracy(val_batches)
        
        # Qualitative evaluation
        print("Generating sample spells...")
        sample_spells = self.generate_sample_spells(num_samples, temperature)
        
        # Format compliance check
        format_scores = []
        for spell in sample_spells:
            compliance = self.check_format_compliance(spell)
            format_scores.append(compliance)
        
        # Calculate overall format compliance rate
        total_checks = len(format_scores) * len(format_scores[0]) if format_scores else 0
        passed_checks = sum(sum(checks.values()) for checks in format_scores)
        format_compliance_rate = passed_checks / total_checks if total_checks > 0 else 0.0
        
        results = {
            'perplexity': perplexity,
            'token_accuracy': accuracy,
            'format_compliance_rate': format_compliance_rate,
            'sample_spells': sample_spells,
            'format_checks': format_scores,
            'num_parameters': self.model.get_num_params()
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """Print a formatted evaluation report"""
        print("\n" + "="*60)
        print("📊 SPELLFORGE MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📈 QUANTITATIVE METRICS:")
        print(f"  • Perplexity:      {results['perplexity']:.2f}")
        print(f"  • Token Accuracy:  {results['token_accuracy']:.1%}")
        print(f"  • Format Compliance: {results['format_compliance_rate']:.1%}")
        print(f"  • Model Parameters: {results['num_parameters']:,}")
        
        print(f"\n📝 SAMPLE GENERATED SPELLS:")
        for i, spell in enumerate(results['sample_spells'], 1):
            print(f"\n--- Sample {i} ---")
            print(spell[:200] + "..." if len(spell) > 200 else spell)
        
        print(f"\n✅ FORMAT COMPLIANCE BREAKDOWN:")
        if results['format_checks']:
            check_names = list(results['format_checks'][0].keys())
            for check in check_names:
                passed = sum(1 for checks in results['format_checks'] if checks[check])
                total = len(results['format_checks'])
                print(f"  • {check.replace('_', ' ').title()}: {passed}/{total} ({passed/total:.1%})")
        
        print("\n" + "="*60)
        
        # Interpretation
        print("\n💡 INTERPRETATION:")
        if results['perplexity'] < 50:
            print("  • Low perplexity - model predictions are confident")
        elif results['perplexity'] > 200:
            print("  • High perplexity - model may be undertrained")
        else:
            print("  • Moderate perplexity - reasonable uncertainty")
        
        if results['format_compliance_rate'] > 0.7:
            print("  • Good format compliance - model learned spell structure")
        else:
            print("  • Poor format compliance - may need more training")


def quick_test_untrained_model(model, charToId, idToChar, device='cpu'):
    """
    Quick test to see what an untrained model generates (should be nonsense).
    """
    print("\n🧪 TESTING UNTRAINED MODEL (should generate nonsense):")
    
    evaluator = SpellEvaluator(model, charToId, idToChar, device)
    
    prompt = "<<< New Spell Forged >>>\n  Name: "
    prompt_ids = evaluator.encode_prompt(prompt)
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=100,
            temperature=1.0,
            do_sample=True
        )
    
    generated_text = evaluator.decode_tokens(generated_ids)
    print(f"Generated: {generated_text[:150]}...")
    
    return generated_text