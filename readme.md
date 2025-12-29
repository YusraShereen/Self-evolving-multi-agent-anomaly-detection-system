# Self-Evolving Multi-Agent Anomaly Detection System

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



