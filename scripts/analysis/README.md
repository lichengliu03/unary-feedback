# Analysis and Visualization Scripts

This directory contains all data analysis and visualization scripts for the UFO project.

## Overview

These scripts process evaluation results and generate figures for papers, presentations, and documentation.

## Scripts

### Comparison Scripts

**`compare_feedback_mechanisms_comprehensive.py`**
- Compares different feedback mechanisms (Normal, Critique, No Feedback)
- Generates 4-panel comprehensive analysis figure
- Outputs to `results/feedback_comparison/`

**`compare_1turn_vs_5turn_training.py`**
- Compares models trained with different max turns
- Shows impact of multi-turn training
- Analyzes performance across different k values

**`compare_base_vs_trained_models.py`**
- Compares base models with UFO-trained models
- Shows improvement from training
- Highlights multi-turn benefits

### Plotting Scripts

**`plot_conditional_success_rates_by_training_steps.py`**
- Plots conditional success rates over training steps
- Shows learning progress
- Useful for analyzing training dynamics

**`plot_single_experiment_passk.py`**
- Plots Pass@k curves for a single experiment
- Shows success rate vs number of attempts
- Useful for quick experiment visualization

**`plot_independent_passk.py`**
- Plots independent Pass@k results
- Compares with multi-turn results
- Shows baseline performance

**`plot_1turn_vs_5turn_passk.py`**
- Specialized comparison of 1-turn vs 5-turn training
- Shows Pass@k curves side by side

## Usage

### Basic Usage

```bash
# Compare feedback mechanisms
python analysis/compare_feedback_mechanisms_comprehensive.py

# Plot single experiment
python analysis/plot_single_experiment_passk.py \
    --input results/my_experiment/metrics.json \
    --output results/my_experiment/passk_curve.png
```

### Input Format

Most scripts expect JSON files with evaluation results:

```json
{
  "pass_at_k": {
    "k1": 0.489,
    "k2": 0.554,
    "k3": 0.592,
    "k4": 0.609,
    "k5": 0.615
  },
  "conditional_success": {
    "given_k1_failed": 0.132,
    "given_k2_failed": 0.070,
    ...
  }
}
```

### Output Location

All scripts output to `results/[experiment_name]/`:
- Figures: `.png` files
- Processed data: `.json` files

## Adding New Analysis Scripts

### Template

```python
#!/usr/bin/env python3
"""
my_analysis.py - Brief description

Usage:
    python analysis/my_analysis.py --input <file> --output <file>
"""

import argparse
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="My analysis")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output figure path")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting code ...

    # Save
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {args.output}")

if __name__ == "__main__":
    main()
```

### Best Practices

**Figure Quality**:
- Use 300 DPI for publications
- Set figure size appropriately (10x6 inches is good default)
- Use `bbox_inches='tight'` to avoid clipping
- Save as PNG for web, PDF for papers

**Colors**:
- Use colorblind-friendly palettes
- Be consistent across figures
- Add legends and labels

**Code Style**:
- Add docstrings
- Use argparse for command-line arguments
- Handle errors gracefully
- Print progress messages

## Common Workflows

### Analyzing New Experiment

1. **Run evaluation** and save results to JSON
2. **Generate figures**:
   ```bash
   python analysis/plot_single_experiment_passk.py \
       --input results/my_exp/metrics.json \
       --output results/my_exp/passk.png
   ```
3. **Compare with baselines**:
   ```bash
   python analysis/compare_base_vs_trained_models.py \
       --baseline results/baseline/metrics.json \
       --trained results/my_exp/metrics.json \
       --output results/comparison/
   ```

### Creating Paper Figures

1. Generate all figures with consistent style
2. Use high DPI (300+)
3. Ensure readable font sizes
4. Add clear labels and legends
5. Save both PNG and PDF versions

### Batch Processing

```bash
# Process multiple experiments
for exp in exp1 exp2 exp3; do
    python analysis/plot_single_experiment_passk.py \
        --input results/$exp/metrics.json \
        --output results/$exp/passk.png
done
```

## Dependencies

Required packages:
- `matplotlib` - Plotting
- `numpy` - Numerical operations
- `pandas` - Data manipulation (optional)
- `seaborn` - Statistical visualization (optional)

Install with:
```bash
pip install matplotlib numpy pandas seaborn
```

## Troubleshooting

**Import errors**:
- Ensure you're running from project root
- Check PYTHONPATH includes project directory

**Figure not saving**:
- Check output directory exists
- Verify write permissions
- Ensure sufficient disk space

**Data format errors**:
- Verify JSON structure matches expected format
- Check for missing keys
- Validate data types

## Related Documentation

- **Evaluation**: See `eval/README.md` for evaluation workflows
- **Results**: See `results/README.md` for result organization
- **Tools**: See `tools/README.md` for data processing tools

## Future Improvements

Potential additions:
- Interactive visualizations (Plotly, Bokeh)
- Statistical significance testing
- Automated report generation
- Dashboard for real-time monitoring
