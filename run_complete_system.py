
import asyncio
import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Phase 1 imports
from mcp_server import create_mcp_server
from dataset import BrainMRIDataset, get_transforms
from phase1_ensemble import Phase1EnsembleSystem
from evaluation import PerformanceEvaluator

# Phase 2 imports
from phase2_vlm_agent import Phase2ExplainerSystem
from grpo_training import prepare_grpo_dataset, setup_grpo_training, train_grpo


def load_qwen_model(model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"):
    """
    Load Qwen3-VL model with LoRA for GRPO training
    
    Args:
        model_name: Model identifier
        
    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    from unsloth import FastVisionModel
    from transformers import BitsAndBytesConfig
    
    print(f"\n Loading Qwen3-VL model: {model_name}")
    
    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=16384,
        quantization_config=bnb_config,
        fast_inference=False,
        gpu_memory_utilization=0.8,
        offload_folder="./offload"
    )
    
    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        use_gradient_checkpointing="unsloth"
    )
    
    print(f" Model loaded successfully")
    return model, tokenizer


async def run_complete_system(config: dict):
    """
    Run complete two-phase system
    
    Args:
        config: Configuration dictionary
    """
    
    print("\n" + "="*70)
    print("BRAIN MRI COMPLETE MULTI-AGENT SYSTEM")
    print("Phase 1: ViT Autoencoder Ensemble")
    print("Phase 2: VLM Explainer with GRPO")
    print("="*70 + "\n")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    print("Step 1: Initialization...")
    mcp_server = create_mcp_server()
    
    # Load datasets
    print("\nStep 2: Loading datasets...")
    train_transform, test_transform = get_transforms()
    
    train_dataset = BrainMRIDataset(
        config['data_root'], mode='train', transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['phase1']['batch_size'],
        shuffle=True,
        num_workers=config['phase1']['num_workers'],
        pin_memory=True
    )
    
    # Load test subsets
    test_subsets = config.get('test_subsets', ['T1', 'T2', 'T3', 'T4'])
    test_loaders = {}
    test_datasets = {}
    
    for subset in test_subsets:
        try:
            ds = BrainMRIDataset(
                config['data_root'],
                mode='test',
                test_subset=subset,
                transform=test_transform
            )
            test_loaders[subset] = DataLoader(
                ds,
                batch_size=config['phase1']['batch_size'],
                shuffle=False,
                num_workers=config['phase1']['num_workers'],
                pin_memory=True
            )
            test_datasets[subset] = ds
        except Exception as e:
            print(f"   {subset} not available: {e}")
    
    # ========================================================================
    # PHASE 1: VIT AUTOENCODER ENSEMBLE
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: VIT AUTOENCODER ENSEMBLE")
    print("="*70)
    
    phase1 = Phase1EnsembleSystem(
        num_agents=config['phase1']['num_agents'],
        mcp_server=mcp_server,
        model_name=config['phase1']['model_name']
    )
    
    # Train or load Phase 1
    if config['phase1'].get('load_checkpoint'):
        print(f"\nLoading Phase 1 checkpoint: {config['phase1']['load_checkpoint']}")
        phase1.load_models(config['phase1']['load_checkpoint'])
    else:
        print("\nTraining Phase 1...")
        await phase1.train(
            train_loader,
            epochs=config['phase1']['epochs'],
            save_interval=config['phase1'].get('save_interval', 500)
        )
        phase1.save_models(config['phase1']['checkpoint_dir'])
    
    # Evaluate Phase 1
    print("\nEvaluating Phase 1 on test sets...")
    evaluator = PerformanceEvaluator(output_dir=config['output_dir'] + '/phase1')
    
    phase1_results = {}
    test_set_for_grpo = config.get('grpo_test_set', 'T1')
    
    for test_name in sorted(test_loaders.keys()):
        eval_result = await phase1.evaluate_temporal_subset(
            test_loaders[test_name], test_name
        )
        
        y_true = test_datasets[test_name].labels_array
        y_pred = eval_result['predictions']
        scores = eval_result['scores']
        
        metrics = evaluator.evaluate_predictions(y_true, y_pred, scores, test_name)
        metrics['drift_detection'] = eval_result['drift_detection']
        metrics['threshold'] = eval_result['threshold']
        
        phase1_results[test_name] = eval_result
        
        # Store for GRPO if this is the selected test set
        if test_name == test_set_for_grpo:
            grpo_phase1_results = {
                'flagged_indices': eval_result['drift_detection'].get('flagged_indices', 
                    np.where(eval_result['scores'] > eval_result['threshold'])[0]),
                'scores': eval_result['scores'],
                'threshold': eval_result['threshold']
            }
        
        print(f"\n{test_name} - Phase 1 Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Requires retraining: {eval_result['drift_detection']['requires_retraining']}")
    
    # ========================================================================
    # PHASE 2: VLM EXPLAINER WITH GRPO
    # ========================================================================
    if config['phase2']['enable']:
        print("\n" + "="*70)
        print("PHASE 2: VLM EXPLAINER WITH GRPO")
        print("="*70)
        
        # Initialize Phase 2
        phase2 = Phase2ExplainerSystem(mcp_server)
        
        # Load VLM model
        if config['phase2'].get('load_vlm_checkpoint'):
            print(f"\nLoading VLM from checkpoint: {config['phase2']['load_vlm_checkpoint']}")
            # Load from checkpoint
            from unsloth import FastVisionModel
            model, tokenizer = FastVisionModel.from_pretrained(
                config['phase2']['load_vlm_checkpoint']
            )
        else:
            # Load base model
            model, tokenizer = load_qwen_model(config['phase2']['model_name'])
        
        phase2.load_vlm_model(model, tokenizer)
        
        # GRPO Training
        if config['phase2']['grpo']['enable_training']:
            print("\n" + "="*70)
            print("GRPO TRAINING")
            print("="*70)
            
            # Prepare GRPO dataset
            grpo_dataset = prepare_grpo_dataset(
                test_datasets[test_set_for_grpo],
                grpo_phase1_results,
                tokenizer,
                num_samples=config['phase2']['grpo']['num_samples']
            )
            
            # Setup GRPO trainer
            trainer = setup_grpo_training(
                model,
                tokenizer,
                grpo_dataset,
                output_dir=config['phase2']['grpo']['checkpoint_dir'],
                num_train_epochs=config['phase2']['grpo']['epochs'],
                learning_rate=config['phase2']['grpo']['learning_rate']
            )
            
            # Train
            train_grpo(trainer, save_path=config['phase2']['grpo']['final_model_dir'])
            
            # Reload trained model
            print("\nReloading GRPO-trained model...")
            del model, tokenizer
            torch.cuda.empty_cache()
            
            from unsloth import FastVisionModel
            model, tokenizer = FastVisionModel.from_pretrained(
                config['phase2']['grpo']['final_model_dir']
            )
            phase2.load_vlm_model(model, tokenizer)
        
        # Phase 2 Evaluation
        print("\n" + "="*70)
        print("PHASE 2 EVALUATION")
        print("="*70)
        
        phase2_evaluator = PerformanceEvaluator(
            output_dir=config['output_dir'] + '/phase2'
        )
        
        for test_name in sorted(test_loaders.keys()):
            print(f"\nEvaluating {test_name} with Phase 2...")
            
            # Get flagged samples from Phase 1
            phase1_result = phase1_results[test_name]
            flagged_indices = np.where(
                phase1_result['scores'] > phase1_result['threshold']
            )[0]
            
            if len(flagged_indices) > 0:
                # Analyze with Phase 2
                phase2_result = await phase2.analyze_flagged_samples(
                    flagged_indices,
                    test_datasets[test_name],
                    phase1_result['scores'],
                    phase1_result['threshold']
                )
                
                # Evaluate refined predictions
                y_true = test_datasets[test_name].labels_array
                refined_predictions = (
                    phase2_result['refined_scores'] > phase1_result['threshold']
                ).astype(int)
                
                metrics = phase2_evaluator.evaluate_predictions(
                    y_true, refined_predictions,
                    phase2_result['refined_scores'],
                    test_name
                )
                
                print(f"\n{test_name} - Phase 2 Results:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                print(f"  Mean Reward: {np.mean(phase2_result['rewards']):.4f}")
                
                # Save explanations
                import json
                explanation_file = Path(config['output_dir']) / 'phase2' / f'explanations_{test_name}.json'
                explanation_file.parent.mkdir(parents=True, exist_ok=True)
                with open(explanation_file, 'w') as f:
                    json.dump(phase2_result['explanations'], f, indent=2)
                print(f"  ✓ Explanations saved: {explanation_file}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SYSTEM COMPLETE")
    print("="*70)
    
    print("\n MCP Server Statistics:")
    stats = mcp_server.get_tool_stats()
    for tool_name, tool_stats in stats.items():
        print(f"  {tool_name}: {tool_stats['usage_count']} calls")
    
    print(f"\n Results saved to: {config['output_dir']}")
    print("="*70 + "\n")
    
    return phase1, phase2 if config['phase2']['enable'] else None


def load_config(config_path: str) -> dict:
    """Load configuration from JSON"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Complete Brain MRI Multi-Agent System'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_complete.json',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f" Loaded configuration from {args.config}")
    else:
        print(f" Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'data_root': '/path/to/data',
            'output_dir': './results/complete',
            'test_subsets': ['T1', 'T2', 'T3', 'T4'],
            'grpo_test_set': 'T1',
            'phase1': {
                'num_agents': 3,
                'model_name': 'vit_tiny_patch16_224',
                'epochs': 3000,
                'batch_size': 32,
                'num_workers': 2,
                'checkpoint_dir': './checkpoints/phase1_final',
                'load_checkpoint': None,
                'save_interval': 500
            },
            'phase2': {
                'enable': True,
                'model_name': 'unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit',
                'load_vlm_checkpoint': None,
                'grpo': {
                    'enable_training': True,
                    'epochs': 3,
                    'num_samples': 100,
                    'learning_rate': 5e-5,
                    'checkpoint_dir': './checkpoints/grpo',
                    'final_model_dir': './models/grpo_final'
                }
            }
        }
    
    # Validate
    if not Path(config['data_root']).exists():
        print(f"\n Error: Data root not found: {config['data_root']}")
        print("Please update data_root in config")
        return
    
    # Run system
    try:
        asyncio.run(run_complete_system(config))
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()