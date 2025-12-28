
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from PIL import Image
from tqdm.auto import tqdm

from phase2_vlm_agent import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def format_reward_function(response: str) -> float:
    """
    Reward for following the correct format:
    <REASONING>...</REASONING><SOLUTION>score</SOLUTION>
    
    Args:
        response: VLM response
        
    Returns:
        reward: 0.0 to 1.0
    """
    has_reasoning = REASONING_START in response and REASONING_END in response
    has_solution = SOLUTION_START in response and SOLUTION_END in response
    
    if has_reasoning and has_solution:
        return 1.0
    elif has_reasoning or has_solution:
        return 0.5
    else:
        return 0.0


def accuracy_reward_function(response: str, ground_truth: int,
                             reconstruction_error: float, threshold: float) -> float:
    """
    Reward for accurate abnormality prediction
    
    Args:
        response: VLM response
        ground_truth: 0 (normal) or 1 (abnormal)
        reconstruction_error: Score from Phase 1
        threshold: Decision threshold
        
    Returns:
        reward: 0.0 to 1.0
    """
    try:
        # Extract solution score
        solution_start = response.find(SOLUTION_START)
        solution_end = response.find(SOLUTION_END)
        
        if solution_start == -1 or solution_end == -1:
            return 0.0
        
        solution_text = response[solution_start + len(SOLUTION_START):solution_end].strip()
        
        # Parse as float
        try:
            predicted_score = float(solution_text)
        except:
            # Fallback: keyword detection
            if any(word in solution_text.lower() for word in ["abnormal", "patholog", "anomal"]):
                predicted_score = 1.0
            else:
                predicted_score = 0.0
        
        # Clip to [0, 1]
        predicted_score = max(0.0, min(1.0, predicted_score))
        
        # Calculate reward
        error = abs(predicted_score - ground_truth)
        reward = 1.0 - error
        
        return max(0.0, reward)
        
    except Exception as e:
        return 0.0


def reasoning_quality_reward(response: str) -> float:
    """
    Reward for quality of reasoning (length, medical terms, structure)
    
    Args:
        response: VLM response
        
    Returns:
        reward: 0.0 to 1.0
    """
    try:
        reasoning_start = response.find(REASONING_START)
        reasoning_end = response.find(REASONING_END)
        
        if reasoning_start == -1 or reasoning_end == -1:
            return 0.0
        
        reasoning = response[reasoning_start + len(REASONING_START):reasoning_end].strip()
        
        # Minimum length check
        if len(reasoning) < 20:
            return 0.0
        
        # Check for medical/anatomical terms
        medical_terms = [
            "tissue", "contrast", "symmetry", "mass", "signal", "lesion",
            "ventricle", "cortex", "white matter", "gray matter", "hemisphere",
            "abnormality", "pathology", "anatomy", "structure", "intensity",
            "atrophy", "edema", "hemorrhage", "infarct", "tumor","glioma","meningioma","pituitary"
        ]
        
        term_count = sum(1 for term in medical_terms if term.lower() in reasoning.lower())
        
        # Length reward (up to 0.5)
        length_reward = min(0.5, len(reasoning) / 200)
        
        # Medical term reward (up to 0.5)
        term_reward = min(0.5, term_count * 0.1)
        
        return length_reward + term_reward
        
    except:
        return 0.0


def combined_reward_function(response: str, ground_truth: int,
                             reconstruction_error: float, threshold: float) -> float:
    """
    Combined reward function with weighted components
    
    Args:
        response: VLM response
        ground_truth: True label
        reconstruction_error: Phase 1 score
        threshold: Decision threshold
        
    Returns:
        total_reward: Weighted combination
    """
    format_r = format_reward_function(response)
    accuracy_r = accuracy_reward_function(response, ground_truth, reconstruction_error, threshold)
    quality_r = reasoning_quality_reward(response)
    
    # Weights: format (0.2), accuracy (0.6), quality (0.2)
    total_reward = 0.2 * format_r + 0.6 * accuracy_r + 0.2 * quality_r
    
    return total_reward


# ============================================================================
# GRPO DATASET PREPARATION
# ============================================================================

@dataclass
class GRPOSample:
    """Single sample for GRPO training"""
    image_path: str
    reconstruction_error: float
    threshold: float
    ground_truth: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor


def prepare_grpo_dataset(test_dataset, phase1_results: Dict,
                        tokenizer, num_samples: int = 100) -> List[Dict]:
    """
    Prepare dataset for GRPO training from flagged samples
    
    Args:
        test_dataset: Test dataset
        phase1_results: Results from Phase 1 (contains flagged_indices, scores, threshold)
        tokenizer: Qwen3 tokenizer
        num_samples: Number of samples to use
        
    Returns:
        grpo_data: List of prepared samples
    """
    print(f"\n Preparing GRPO dataset...")
    
    flagged_indices = phase1_results['flagged_indices']
    scores = phase1_results['scores']
    threshold = phase1_results['threshold']
    
    # Select samples: mix of flagged and normal
    selected_indices = []
    
    # Take flagged samples
    n_flagged = min(len(flagged_indices), num_samples // 2)
    selected_indices.extend(flagged_indices[:n_flagged])
    
    # Take normal samples for balance
    normal_indices = [i for i in range(len(scores)) if scores[i] <= threshold]
    n_normal = min(len(normal_indices), num_samples - n_flagged)
    selected_indices.extend(normal_indices[:n_normal])
    
    print(f"  Selected {len(selected_indices)} samples:")
    print(f"    Flagged: {n_flagged}")
    print(f"    Normal: {n_normal}")
    
    grpo_data = []
    
    for idx in tqdm(selected_indices, desc="Preparing samples"):
        image_path = test_dataset.samples[idx]
        reconstruction_error = scores[idx]
        ground_truth = test_dataset.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create prompt
        text_content = (
            f"You are an expert radiologist. This brain MRI has been evaluated "
            f"with a reconstruction error of {reconstruction_error:.4f} "
            f"against a threshold of {threshold:.4f}.\n\n"
            f"Analyze this scan carefully and provide:\n"
            f"1. Your detailed clinical reasoning between {REASONING_START} and {REASONING_END}\n"
            f"2. Final abnormality score (0.0=normal, 1.0=abnormal) between "
            f"{SOLUTION_START} and {SOLUTION_END}\n\n"
            f"Consider: tissue contrast, symmetry, mass effect, signal abnormalities."
        )
        
        # Apply chat template
        prompt_chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_content}
                ]
            }
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_chat, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            image,
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Create sample dictionary
        sample = {
            'image_path': image_path,
            'reconstruction_error': float(reconstruction_error),
            'threshold': float(threshold),
            'ground_truth': int(ground_truth),
            'prompt': text_content,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0)
        }
        
        grpo_data.append(sample)
    
    n_abnormal = sum(1 for s in grpo_data if s['ground_truth'] == 1)
    n_normal = len(grpo_data) - n_abnormal
    
    print(f"\n✓ GRPO dataset prepared:")
    print(f"    Total samples: {len(grpo_data)}")
    print(f"    Abnormal: {n_abnormal}")
    print(f"    Normal: {n_normal}")
    
    return grpo_data


# ============================================================================
# GRPO TRAINING SETUP
# ============================================================================

def setup_grpo_training(model, tokenizer, grpo_dataset: List[Dict],
                       output_dir: str = "./grpo_checkpoints",
                       num_train_epochs: int = 3,
                       learning_rate: float = 5e-5):
    """
    Setup GRPO trainer with custom reward functions
    
    Args:
        model: Qwen3-VL model
        tokenizer: Qwen3 tokenizer
        grpo_dataset: Prepared GRPO dataset
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        trainer: Configured GRPO trainer
    """
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import vLLMSamplingParams
    
    print(f"\ Setting up GRPO training...")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    # Define reward functions for TRL
    def format_reward(completions, **kwargs):
        return [format_reward_function(c) for c in completions]
    
    def accuracy_reward(completions, prompts, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            sample = grpo_dataset[i % len(grpo_dataset)]
            reward = accuracy_reward_function(
                completion,
                sample['ground_truth'],
                sample['reconstruction_error'],
                sample['threshold']
            )
            rewards.append(reward)
        return rewards
    
    def quality_reward(completions, **kwargs):
        return [reasoning_quality_reward(c) for c in completions]
    
    # GRPO Configuration
    max_seq_length = 1024
    max_prompt_length = 220
    max_completion_length = max_seq_length - max_prompt_length
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        
        # GRPO specific
        temperature=1.0,
        num_generations=2,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        
        # Logging
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        
        # KL penalty
        beta=0.04,
        
        # vLLM sampling
        vllm_sampling_params=vLLMSamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=256,
            min_p=0.1,
            seed=3407
        ),
        
        report_to="none"
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward, quality_reward],
        args=training_args,
        train_dataset=grpo_dataset
    )
    
    print(f" GRPO trainer configured")
    print(f"  Reward functions: Format + Accuracy + Quality")
    
    return trainer


def train_grpo(trainer, save_path: str = "./grpo_final_model"):
    """
    Run GRPO training
    
    Args:
        trainer: Configured GRPO trainer
        save_path: Path to save final model
        
    Returns:
        training_stats: Training statistics
    """
    print(f"\n Starting GRPO training...")
    
    # Train
    trainer.train()
    
    # Save model
    Path(save_path).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(save_path)
    trainer.processing_class.save_pretrained(save_path)
    
    print(f"\n GRPO training complete")
    print(f"  Model saved to: {save_path}")
    
    return {
        'training_complete': True,
        'save_path': save_path
    }
