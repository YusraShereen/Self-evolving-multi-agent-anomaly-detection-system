
import numpy as np
import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class PerformanceEvaluator:
    """Comprehensive performance evaluation"""
    
    def __init__(self, output_dir: str = "./results"):
        """
        Args:
            output_dir: Directory to save results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Performance Evaluator initialized")
        print(f"  Output directory: {self.output_dir}")
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           scores: np.ndarray, test_name: str) -> Dict:
        """
        Compute all metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            scores: Anomaly scores
            test_name: Name of test set
            
        Returns:
            results: Dictionary of metrics
        """
        # Basic metrics
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        acc = (y_true == y_pred).mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity and Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results = {
            'test_name': test_name,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, test_name: str, 
                             save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            test_name: Name of test set
            save_path: Optional custom save path
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {test_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'confusion_matrix_{test_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Confusion matrix saved: {save_path}")
    
    def plot_score_distribution(self, scores: np.ndarray, y_true: np.ndarray,
                               threshold: float, test_name: str,
                               save_path: str = None):
        """
        Plot score distribution for normal vs abnormal
        
        Args:
            scores: Anomaly scores
            y_true: Ground truth labels
            threshold: Decision threshold
            test_name: Name of test set
            save_path: Optional custom save path
        """
        plt.figure(figsize=(10, 6))
        
        normal_scores = scores[y_true == 0]
        abnormal_scores = scores[y_true == 1]
        
        plt.hist(normal_scores, bins=30, alpha=0.6, label='Normal', 
                color='green', edgecolor='black')
        plt.hist(abnormal_scores, bins=30, alpha=0.6, label='Abnormal', 
                color='red', edgecolor='black')
        plt.axvline(threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Score Distribution - {test_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f'score_distribution_{test_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Score distribution saved: {save_path}")
    
    def generate_classification_report(self, y_true: np.ndarray,
                                       y_pred: np.ndarray, test_name: str):
        """
        Generate detailed classification report
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            test_name: Name of test set
        """
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Abnormal'],
            output_dict=True,
            zero_division=0
        )
        
        # Save as JSON
        report_path = self.output_dir / f'classification_report_{test_name}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print to console
        print(f"\n{'='*60}")
        print(f"Classification Report - {test_name}")
        print(f"{'='*60}")
        print(classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Abnormal'],
            zero_division=0
        ))
        
        print(f"  ✓ Classification report saved: {report_path}")
    
    def plot_temporal_comparison(self, temporal_results: Dict, 
                                save_path: str = None):
        """
        Plot metrics across temporal test sets
        
        Args:
            temporal_results: Dictionary mapping test names to metrics
            save_path: Optional custom save path
        """
        test_sets = sorted(temporal_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                  'specificity', 'sensitivity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [temporal_results[ts][metric] for ts in test_sets]
            
            axes[idx].plot(test_sets, values, marker='o', linewidth=2, 
                          markersize=8, color='steelblue')
            axes[idx].set_title(metric.replace('_', ' ').title(), 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Test Set', fontsize=10)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1.05])
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', 
                             ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Temporal Performance Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'temporal_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Temporal comparison saved: {save_path}")
    
    def save_results_table(self, temporal_results: Dict):
        """
        Save results as CSV and LaTeX tables
        
        Args:
            temporal_results: Dictionary mapping test names to metrics
        """
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(temporal_results).T
        
        # Select key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                      'specificity', 'sensitivity']
        
        if all(m in df.columns for m in key_metrics):
            df_export = df[key_metrics]
        else:
            df_export = df
        
        # Save as CSV
        csv_path = self.output_dir / 'results_table.csv'
        df_export.to_csv(csv_path, float_format='%.4f')
        
        # Save as LaTeX
        latex_table = df_export.to_latex(
            float_format='%.4f',
            caption='Performance Metrics Across Temporal Test Sets',
            label='tab:results'
        )
        latex_path = self.output_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
    
    def save_json(self, data: Dict, filename: str):
        """
        Save data as JSON
        
        Args:
            data: Data to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   JSON saved: {output_path}")

