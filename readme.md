# MCP-based Multi-Agent Anomaly Detection System

##  Project Structure

```
multi_agent_system/
│
── PHASE 1
|── mcp_server.py              # MCP Protocol & Tools
├── vit_autoencoder.py         # ViT Architecture
├── evaluation.py              # Metrics & Visualization
├── main.py                    # Phase 1 Orchestrator
│
├──  PHASE 2
│   ├── phase2_vlm_agent.py        # VLM Explainer Agent
│   ├── grpo_training.py           # GRPO Training Module
│   └── run_complete_system.py    # Complete Pipeline
│
├── README.md                  # This file
│
│
└── data/                      # Dataset 
    └── Transformed_DS/
        ├── Train/             # Normal samples only
        └── Test/
            ├── T1/            # Temporal subset 1
            │   ├── 0/         # Normal
            │   └── 1/         # Abnormal
            ├── T2/            # Temporal subset 2
            │   ├── 0/         # Normal
            │   └── 1/         # Abnormal
            ├── T3/            # Temporal subset 3
            │   ├── 0/         # Normal
            │   └── 1/         # Abnormal
            └── T4/            # Temporal subset 4
            │   ├── 0/         # Normal
            │   └── 1/         # Abnormal
```

##  Module Architecture

### 1. **mcp_server.py** - MCP Protocol Implementation
- `MCPServer`: Central tool orchestrator
- `ThresholdOptimizationTool`: 3-sigma threshold computation
- `EnsembleConsensusTool`: Multi-agent consensus
- `DriftDetectionTool`: Distribution drift detection

**Key Features:**
- Tool registry with execution logging
- Standardized result format
- Usage statistics tracking

### 2. **vit_autoencoder.py** - ViT Architecture
- `ViTAutoencoder`: Vision Transformer autoencoder
- Patch-wise reconstruction
- Per-patch MSE error computation

**Architecture:**
- Encoder: Pre-trained ViT 
- Decoder: 2-layer MLP
- Patch size: 16×16
- Image size: 224×224


**Workflow:**
1. Train 3 agents independently
2. Compute consensus scores
3. Calculate 3σ threshold from training
4. Test with drift detection
5. Flag datasets requiring retraining

### 6. **evaluation.py** - Performance Evaluation
- `PerformanceEvaluator`: Comprehensive metrics
- Confusion matrices
- Score distributions
- Temporal comparisons

**Metrics:**
- Accuracy, Precision, Recall, F1
- Specificity, Sensitivity
- Classification reports

### 7. **main.py** - Phase 1 Orchestrator
- Complete pipeline execution
- Configuration management
- Command-line interface

### 8. **run_complete_system.py** - Complete Pipeline

### Configuration

Create `config.json`:

```json
{
  "data_root": "/path/to/data",
  "num_agents": 3,
  "epochs": 3000,
  "batch_size": 32,
  "num_workers": 2,
  "model_name": "vit_tiny_patch16_224",
  "output_dir": "./results/phase1",
  "checkpoint_dir": "./checkpoints/phase1_final",
  "save_interval": 500,
  "test_subsets": ["T1", "T2", "T3", "T4"]
}
```

### Training

```bash
# Train from scratch
python main.py --config config.json

# Override data path
python main.py --config config.json --data-root /new/path/to/data

# Override epochs
python main.py --config config.json --epochs 1000

# Load from checkpoint
python main.py --config config.json --load-checkpoint ./checkpoints/phase1_epoch1500
```
### Output Files

1. **Checkpoints** (`checkpoints/phase1_final/`)
   - `agent_0.pt`, `agent_1.pt`, `agent_2.pt`
   - `statistics.json`

2. **Visualizations** (`results/phase1/`)
   - Confusion matrices (PNG)
   - Score distributions (PNG)
   - Temporal comparison (PNG)

3. **Reports** (`results/phase1/`)
   - `results_table.csv`
   - `results_table.tex`
   - `complete_results.json`
   - Classification reports (JSON)

## Expected Timeline & Resources

### Phase 1 Training (3000 epochs)
- **Time**: ~12 hours (GPU)
- **Memory**: 8-16 GB VRAM
- **Output**: 3 trained agents + statistics

### GRPO Training (3 epochs)
- **Time**: ~4-6 hours (GPU)
- **Memory**: 16-24 GB VRAM (4-bit quantization)
- **Output**: Fine-tuned VLM model

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0
pandas>=2.0.0
unsloth
```

## 🎓 Citation

If you use this code for research, please cite:

```bibtex
@software{brain_mri_mcp,
  title={Brain MRI Multi-Agent Anomaly Detection with MCP Protocol},
  author={Yusra Shereen},
  year={2025}}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact yusrashereen@gmail.com.

