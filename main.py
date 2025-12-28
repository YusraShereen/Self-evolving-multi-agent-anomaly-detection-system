
import asyncio
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader

from mcp_server import create_mcp_server
from dataset import BrainMRIDataset, get_transforms
from phase1_ensemble import Phase1EnsembleSystem
from evaluation import PerformanceEvaluator


async def run_phase1_system(config: dict):
    """
    Main execution pipeline for Phase 1
    
    Args:
        config: Configuration dictionary
    """
    
    print("\n" + "="*70)
    print("BRAIN MRI MULTI-AGENT SYSTEM - PHASE 1")
    print("="*70)
    print(f"Configuration:")
    print(f"  • Data root: {config['data_root']}")
    print(f"  • Agents: {config['num_agents']}")
    print(f"  • Epochs: {config['epochs']}")
    print(f"  • Batch size: {config['batch_size']}")
    print(f"  • Model: {config['model_name']}")
    print("="*70 + "\n")
    
    # ========================================================================
    # 1. INITIALIZE MCP SERVER
    # ========================================================================
    print("Step 1: Initializing MCP Server...")
    mcp_server = create_mcp_server()
    
    # ========================================================================
    # 2. LOAD DATASETS
    # ========================================================================
    print("\nStep 2: Loading datasets...")
    train_transform, test_transform = get_transforms()
    
    # Training dataset
    train_dataset = BrainMRIDataset(
        config['data_root'], 
        mode='train', 
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Temporal test subsets
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
                batch_size=config['batch_size'], 
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            test_datasets[subset] = ds
        except Exception as e:
            print(f"  {subset} not available: {e}")
    
    # ========================================================================
    # 3. INITIALIZE PHASE 1 SYSTEM
    # ========================================================================
    print(f"\nStep 3: Initializing Phase 1 system...")
    phase1 = Phase1EnsembleSystem(
        num_agents=config['num_agents'],
        mcp_server=mcp_server,
        model_name=config['model_name']
    )
    
    # ========================================================================
    # 4. TRAIN OR LOAD MODEL
    # ========================================================================
    if config.get('load_checkpoint'):
        print(f"\nStep 4: Loading checkpoint from {config['load_checkpoint']}...")
        phase1.load_models(config['load_checkpoint'])
    else:
        print(f"\nStep 4: Training Phase 1 agents...")
        await phase1.train(
            train_loader, 
            epochs=config['epochs'],
            save_interval=config.get('save_interval', 500)
        )
        
        # Save trained models
        checkpoint_dir = config.get('checkpoint_dir', './checkpoints/phase1_final')
        phase1.save_models(checkpoint_dir)
    
    # ========================================================================
    # 5. TEMPORAL EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("Step 5: TEMPORAL EVALUATION")
    print("="*70)
    
    evaluator = PerformanceEvaluator(output_dir=config['output_dir'])
    all_results = {}
    retraining_required = {}
    
    for test_name in sorted(test_loaders.keys()):
        test_loader = test_loaders[test_name]
        test_dataset = test_datasets[test_name]
        
        # Evaluate with drift detection
        eval_result = await phase1.evaluate_temporal_subset(test_loader, test_name)
        
        # Get ground truth
        y_true = test_dataset.labels_array
        y_pred = eval_result['predictions']
        scores = eval_result['scores']
        
        # Compute metrics
        metrics = evaluator.evaluate_predictions(y_true, y_pred, scores, test_name)
        
        # Add drift information
        metrics['drift_detection'] = eval_result['drift_detection']
        metrics['agreement_score'] = eval_result['agreement']
        metrics['threshold'] = eval_result['threshold']
        
        all_results[test_name] = metrics
        retraining_required[test_name] = eval_result['drift_detection']['requires_retraining']
        
        # Generate visualizations
        evaluator.plot_confusion_matrix(
            np.array(metrics['confusion_matrix']), 
            test_name
        )
        evaluator.plot_score_distribution(
            scores, y_true, eval_result['threshold'], test_name
        )
        evaluator.generate_classification_report(y_true, y_pred, test_name)
        
        # Print summary
        print(f"\n{test_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Retraining required: {retraining_required[test_name]}")
    
    # ========================================================================
    # 6. SUMMARY AND EXPORT
    # ========================================================================
    print("\n" + "="*70)
    print("Step 6: SUMMARY AND EXPORT")
    print("="*70)
    
    # Temporal comparison plot
    evaluator.plot_temporal_comparison(all_results)
    
    # Export tables
    evaluator.save_results_table(all_results)
    
    # Save complete results
    evaluator.save_json(all_results, 'complete_results.json')
    
    # Print summary table
    print(f"\n{'Test':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Retrain':<10}")
    print("-" * 70)
    for name, metrics in sorted(all_results.items()):
        retrain = "Yes" if retraining_required[name] else "No"
        print(f"{name:<6} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
              f"{metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f} {retrain:<10}")
    
    # Print MCP statistics
    print("\n" + "="*70)
    print("MCP SERVER STATISTICS")
    print("="*70)
    stats = mcp_server.get_tool_stats()
    for tool_name, tool_stats in stats.items():
        print(f"  {tool_name}: {tool_stats['usage_count']} calls ({tool_stats['category']})")
    
    print("\n PHASE 1 COMPLETE!")
    print(f"Results saved to: {config['output_dir']}")
    print("="*70 + "\n")
    
    return phase1, all_results, evaluator, mcp_server


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Brain MRI Multi-Agent System - Phase 1'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        help='Override data root path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        help='Load from checkpoint instead of training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f" Loaded configuration from {args.config}")
    else:
        print(f" Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'data_root': '/path/to/data',
            'num_agents': 3,
            'epochs': 3000,
            'batch_size': 32,
            'num_workers': 2,
            'model_name': 'vit_tiny_patch16_224',
            'output_dir': './results/phase1',
            'checkpoint_dir': './checkpoints/phase1_final',
            'save_interval': 500,
            'test_subsets': ['T1', 'T2', 'T3', 'T4']
        }
    
    # Override with command-line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.epochs:
        config['epochs'] = args.epochs
    if args.load_checkpoint:
        config['load_checkpoint'] = args.load_checkpoint
    
    # Validate data root
    if not Path(config['data_root']).exists():
        print(f"\n Error: Data root not found: {config['data_root']}")
        print("Please update the data_root in config.json or use --data-root flag")
        return
    
    # Run Phase 1 system
    try:
        asyncio.run(run_phase1_system(config))
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()