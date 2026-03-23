from pathlib import Path
import numpy as np

# Import all modules
from main_pipeline import ScientificPipeline, ExperimentConfig
from random_topology_generator import TopologyGenerator
from hypothesis_tests import HypothesisTester
from visualize import ScientificVisualizer

# Configure experiment
config = ExperimentConfig(
    n_samples=5000,
    max_homology_dim=2,
    statistical_tests=['wilcoxon', 'permutation'],
    significance_level=0.05,
    random_seed=42
)

# Run complete pipeline
pipeline = ScientificPipeline(config)
results = pipeline.run_complete_analysis()

# Generate publication figures
viz = ScientificVisualizer()
fig = viz.create_publication_figure(
    results, 
    figure_type='comprehensive',
    save_path=Path('figures/main_result.pdf')
)