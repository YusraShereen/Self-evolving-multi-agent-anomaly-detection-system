
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json
from torch.utils.data import DataLoader, ConcatDataset

from mcp_server import MCPServer
from dataset import BrainMRIDataset
from phase1_ensemble import Phase1EnsembleSystem
from evaluation import PerformanceEvaluator


class SelfEvolvingSystem:
    
    def __init__(self, mcp_server: MCPServer, config: Dict):
        """
        Args:
            mcp_server: MCP server instance
            config: Configuration dictionary
        """
        self.mcp_server = mcp_server
        self.config = config
        self.phase1_system = None
        
        # Evolution tracking
        self.evolution_history = []
        self.current_generation = 0
        self.total_retrainings = 0
        
        # Statistics
        self.original_train_mean = None
        self.original_train_std = None
        self.original_threshold = None
        
        print(f"\n{'='*70}")
        print("SELF-EVOLVING LEARNING SYSTEM")
        print("="*70)
        print("Key Features:")
        print("  • Automatic drift detection (3-sigma rule)")
        print("  • Self-triggered retraining")
        print("  • Incremental learning from qualified test sets")
        print("  • Evolution history tracking")
        print("="*70 + "\n")
    
    def initialize_phase1(self, num_agents: int = 3, model_name: str = 'vit_tiny_patch16_224'):
        """Initialize Phase 1 system"""
        self.phase1_system = Phase1EnsembleSystem(
            num_agents=num_agents,
            mcp_server=self.mcp_server,
            model_name=model_name
        )
        print("✓ Phase 1 system initialized")
    
    async def initial_training(self, train_loader: DataLoader, epochs: int = 3000):
        """
        Initial training on original training set (Generation 0)
        
        Args:
            train_loader: Initial training data
            epochs: Number of epochs
        """
        print(f"\n{'='*70}")
        print("GENERATION 0: INITIAL TRAINING")
        print("="*70)
        
        # Train Phase 1
        await self.phase1_system.train(train_loader, epochs=epochs)
        
        # Store original statistics
        self.original_train_mean = self.phase1_system.train_mean
        self.original_train_std = self.phase1_system.train_std
        self.original_threshold = self.phase1_system.train_threshold
        
        # Record in evolution history
        self.evolution_history.append({
            'generation': 0,
            'timestamp': datetime.now().isoformat(),
            'type': 'initial_training',
            'train_mean': float(self.original_train_mean),
            'train_std': float(self.original_train_std),
            'threshold': float(self.original_threshold),
            'epochs': epochs,
            'num_training_samples': len(train_loader.dataset)
        })
        
        print(f"\n✓ Generation 0 complete")
        print(f"  Training mean: {self.original_train_mean:.6f}")
        print(f"  Training std: {self.original_train_std:.6f}")
        print(f"  3σ threshold: {self.original_threshold:.6f}")
    
    async def evaluate_and_decide_retraining(self, test_loader: DataLoader, 
                                             test_name: str) -> Tuple[Dict, bool]:
        """
        Evaluate test set and decide if retraining is needed
        
        Args:
            test_loader: Test data loader
            test_name: Name of test set
            
        Returns:
            results: Evaluation results
            requires_retraining: Boolean flag
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {test_name}")
        print("="*70)
        
        # Evaluate with drift detection
        results = await self.phase1_system.evaluate_temporal_subset(test_loader, test_name)
        
        # Extract drift information
        drift_info = results['drift_detection']
        requires_retraining = drift_info['requires_retraining']
        within_bounds = drift_info['within_bounds']
        test_mean = drift_info['test_mean']
        deviation_sigma = drift_info['deviation_sigma']
        
        print(f"\n Drift Analysis:")
        print(f"  Test mean: {test_mean:.6f}")
        print(f"  Deviation: {deviation_sigma:.2f}σ")
        print(f"  Within 3σ bounds: {within_bounds}")
        print(f"  Out of bounds: {drift_info['out_of_bounds_percentage']:.1%}")
        
        if requires_retraining:
            print(f"\n  RETRAINING REQUIRED!")
            print(f"  Reason: Distribution drift detected")
            print(f"  Test set qualifies for incremental learning")
        else:
            print(f"\n No retraining needed")
            print(f"  Distribution within acceptable bounds")
        
        return results, requires_retraining
    
    async def retrain_with_new_data(self, original_train_dataset: BrainMRIDataset,
                                   new_test_dataset: BrainMRIDataset,
                                   test_name: str,
                                   epochs: int = 1000):
        """
        Retrain system with combined original + new qualified data
        
        Args:
            original_train_dataset: Original training dataset
            new_test_dataset: New test dataset (qualified for retraining)
            test_name: Name of test set
            epochs: Number of retraining epochs
        """
        self.current_generation += 1
        self.total_retrainings += 1
        
        print(f"\n{'='*70}")
        print(f"GENERATION {self.current_generation}: RETRAINING")
        print(f"Incorporating data from: {test_name}")
        print("="*70)
        
        # Combine datasets
        combined_dataset = ConcatDataset([original_train_dataset, new_test_dataset])
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        print(f"\n Combined Training Set:")
        print(f"  Original samples: {len(original_train_dataset)}")
        print(f"  New samples: {len(new_test_dataset)}")
        print(f"  Total samples: {len(combined_dataset)}")
        
        # Store old statistics for comparison
        old_mean = self.phase1_system.train_mean
        old_std = self.phase1_system.train_std
        old_threshold = self.phase1_system.train_threshold
        
        # Retrain Phase 1
        print(f"\n Retraining agents for {epochs} epochs...")
        await self.phase1_system.train(combined_loader, epochs=epochs)
        
        # New statistics
        new_mean = self.phase1_system.train_mean
        new_std = self.phase1_system.train_std
        new_threshold = self.phase1_system.train_threshold
        
        # Record evolution
        evolution_record = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'type': 'retraining',
            'trigger_dataset': test_name,
            'num_training_samples': len(combined_dataset),
            'epochs': epochs,
            'statistics_before': {
                'mean': float(old_mean),
                'std': float(old_std),
                'threshold': float(old_threshold)
            },
            'statistics_after': {
                'mean': float(new_mean),
                'std': float(new_std),
                'threshold': float(new_threshold)
            },
            'changes': {
                'mean_delta': float(new_mean - old_mean),
                'std_delta': float(new_std - old_std),
                'threshold_delta': float(new_threshold - old_threshold)
            }
        }
        
        self.evolution_history.append(evolution_record)
        
        # Print comparison
        print(f"\n Evolution Statistics:")
        print(f"  Generation: {self.current_generation}")
        print(f"\n  Before Retraining:")
        print(f"    Mean: {old_mean:.6f}")
        print(f"    Std: {old_std:.6f}")
        print(f"    Threshold: {old_threshold:.6f}")
        print(f"\n  After Retraining:")
        print(f"    Mean: {new_mean:.6f} (Δ {new_mean - old_mean:+.6f})")
        print(f"    Std: {new_std:.6f} (Δ {new_std - old_std:+.6f})")
        print(f"    Threshold: {new_threshold:.6f} (Δ {new_threshold - old_threshold:+.6f})")
        
        # Save checkpoint
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints')) / f'generation_{self.current_generation}'
        self.phase1_system.save_models(str(checkpoint_dir))
        
        print(f"\n✓ Generation {self.current_generation} complete")
        print(f"  Checkpoint saved: {checkpoint_dir}")
    
    def save_evolution_history(self, output_path: str = './evolution_history.json'):
        """Save complete evolution history"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        evolution_data = {
            'total_generations': self.current_generation,
            'total_retrainings': self.total_retrainings,
            'original_statistics': {
                'mean': float(self.original_train_mean),
                'std': float(self.original_train_std),
                'threshold': float(self.original_threshold)
            },
            'current_statistics': {
                'mean': float(self.phase1_system.train_mean),
                'std': float(self.phase1_system.train_std),
                'threshold': float(self.phase1_system.train_threshold)
            },
            'evolution_history': self.evolution_history
        }
        
        with open(output_file, 'w') as f:
            json.dump(evolution_data, f, indent=2)
        
        print(f"\n✓ Evolution history saved: {output_file}")
    
    def print_evolution_summary(self):
        """Print summary of system evolution"""
        print(f"\n{'='*70}")
        print("EVOLUTION SUMMARY")
        print("="*70)
        print(f"Current Generation: {self.current_generation}")
        print(f"Total Retrainings: {self.total_retrainings}")
        
        print(f"\n Statistical Evolution:")
        print(f"  Original → Current")
        print(f"  Mean:      {self.original_train_mean:.6f} → {self.phase1_system.train_mean:.6f}")
        print(f"  Std:       {self.original_train_std:.6f} → {self.phase1_system.train_std:.6f}")
        print(f"  Threshold: {self.original_threshold:.6f} → {self.phase1_system.train_threshold:.6f}")
        
        if self.total_retrainings > 0:
            print(f"\n Retraining History:")
            for record in self.evolution_history:
                if record['type'] == 'retraining':
                    print(f"  Generation {record['generation']}: {record['trigger_dataset']}")
                    print(f"    Samples added: {record['num_training_samples']}")
                    print(f"    Threshold shift: {record['changes']['threshold_delta']:+.6f}")


async def run_self_evolving_pipeline(config: Dict):
    """
    Complete self-evolving pipeline
    
    Args:
        config: Configuration dictionary
    """
    from dataset import get_transforms
    
    print("\n" + "="*70)
    print("SELF-EVOLVING BRAIN MRI SYSTEM")
    print("="*70 + "\n")
    
    # Initialize
    mcp_server = MCPServer()
    mcp_server.register_tool(ThresholdOptimizationTool())
    mcp_server.register_tool(EnsembleConsensusTool())
    mcp_server.register_tool(DriftDetectionTool())
    
    evolving_system = SelfEvolvingSystem(mcp_server, config)
    evolving_system.initialize_phase1(
        num_agents=config['num_agents'],
        model_name=config['model_name']
    )
    
    # Load datasets
    train_transform, test_transform = get_transforms()
    
    print("Loading datasets...")
    train_dataset = BrainMRIDataset(
        config['data_root'], mode='train', transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Load temporal test sets
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
                ds, batch_size=config['batch_size'],
                shuffle=False, num_workers=config['num_workers']
            )
            test_datasets[subset] = ds
        except Exception as e:
            print(f"   {subset} not available: {e}")
    
    # Initial training (Generation 0)
    await evolving_system.initial_training(
        train_loader,
        epochs=config['initial_epochs']
    )
    
    # Temporal evaluation with automatic retraining
    evaluator = PerformanceEvaluator(output_dir=config['output_dir'])
    all_results = {}
    
    # Store original training dataset for incremental learning
    original_train_dataset = train_dataset
    
    for test_name in sorted(test_loaders.keys()):
        # Evaluate and check for drift
        results, requires_retraining = await evolving_system.evaluate_and_decide_retraining(
            test_loaders[test_name], test_name
        )
        
        # Store results
        y_true = test_datasets[test_name].labels_array
        y_pred = results['predictions']
        scores = results['scores']
        
        metrics = evaluator.evaluate_predictions(y_true, y_pred, scores, test_name)
        metrics['drift_detection'] = results['drift_detection']
        metrics['generation'] = evolving_system.current_generation
        all_results[test_name] = metrics
        
        # Print performance
        print(f"\n{test_name} Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # If retraining required, retrain with this test set
        if requires_retraining and config.get('enable_auto_retraining', True):
            await evolving_system.retrain_with_new_data(
                original_train_dataset,
                test_datasets[test_name],
                test_name,
                epochs=config.get('retraining_epochs', 1000)
            )
            
            # Re-evaluate after retraining
            print(f"\n Re-evaluating {test_name} after retraining...")
            results_after = await evolving_system.phase1_system.evaluate_temporal_subset(
                test_loaders[test_name], test_name
            )
            
            y_pred_after = results_after['predictions']
            metrics_after = evaluator.evaluate_predictions(
                y_true, y_pred_after, results_after['scores'], test_name
            )
            
            print(f"\n Performance Improvement:")
            print(f"  Accuracy: {metrics['accuracy']:.4f} → {metrics_after['accuracy']:.4f} "
                  f"({metrics_after['accuracy'] - metrics['accuracy']:+.4f})")
            print(f"  F1-Score: {metrics['f1_score']:.4f} → {metrics_after['f1_score']:.4f} "
                  f"({metrics_after['f1_score'] - metrics['f1_score']:+.4f})")
            
            all_results[f"{test_name}_after_retraining"] = metrics_after
    
    # Final summary
    evolving_system.print_evolution_summary()
    evolving_system.save_evolution_history(config['output_dir'] + '/evolution_history.json')
    
    # Save final results
    evaluator.save_json(all_results, 'self_evolving_results.json')
    
    print("\n Self-evolving system complete!")
    
    return evolving_system, all_results

