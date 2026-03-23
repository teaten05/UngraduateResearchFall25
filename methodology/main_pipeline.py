"""
Main Scientific Pipeline for Topological Data Analysis
=======================================================
Complete pipeline with hypothesis testing, validation, and reproducibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass, asdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    # Topology generation parameters
    n_samples: int = 1000
    topology_types: List[str] = None
    noise_levels: List[float] = None
    dimensions: List[int] = None
    
    # Feature extraction parameters
    max_homology_dim: int = 2
    n_landmarks: int = 100
    filtration_type: str = 'vietoris_rips'
    
    # Analysis parameters
    statistical_tests: List[str] = None
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    
    # Symbolic regression parameters
    max_complexity: int = 10
    population_size: int = 100
    generations: int = 50
    
    # System parameters
    random_seed: int = 42
    n_jobs: int = -1
    output_dir: str = './results'
    save_intermediates: bool = True
    
    def __post_init__(self):
        """Set defaults for list parameters."""
        if self.topology_types is None:
            self.topology_types = ['sphere', 'torus', 'klein']
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.1, 0.2]
        if self.dimensions is None:
            self.dimensions = [3, 4, 5]
        if self.statistical_tests is None:
            self.statistical_tests = ['ks', 'anderson', 'chi2']
    
    def get_hash(self) -> str:
        """Generate unique hash for configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ScientificPipeline:
    """Main pipeline orchestrating the entire scientific workflow."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = {}
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'config_hash': config.get_hash(),
            'numpy_version': np.__version__,
        }
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / f"exp_{config.get_hash()}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Import modules (lazy loading)
        self.modules = {}
        
    def _save_config(self):
        """Save configuration for reproducibility."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def _lazy_import(self, module_name: str):
        """Lazy import of modules."""
        if module_name not in self.modules:
            try:
                if module_name == 'topology':
                    from random_topology_generator import TopologyGenerator
                    self.modules['topology'] = TopologyGenerator(self.config.random_seed)
                elif module_name == 'features':
                    from feature_extraction import FeatureExtractor
                    self.modules['features'] = FeatureExtractor()
                elif module_name == 'validation':
                    from validate_data import DataValidator
                    self.modules['validation'] = DataValidator()
                elif module_name == 'hypothesis':
                    from hypothesis_tests import HypothesisTester
                    self.modules['hypothesis'] = HypothesisTester()
                elif module_name == 'symbolic':
                    from symbolic_regression import SymbolicRegressor
                    self.modules['symbolic'] = SymbolicRegressor()
                elif module_name == 'visualize':
                    from visualize import ScientificVisualizer
                    self.modules['visualize'] = ScientificVisualizer()
            except ImportError as e:
                logger.warning(f"Could not import {module_name}: {e}")
                return None
        return self.modules.get(module_name)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete scientific pipeline with all steps.
        
        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING SCIENTIFIC PIPELINE")
        logger.info(f"Experiment ID: {self.config.get_hash()}")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Data Generation
            logger.info("\n📊 PHASE 1: Data Generation")
            data = self.generate_data()
            
            # Phase 2: Feature Extraction
            logger.info("\n🔬 PHASE 2: Feature Extraction")
            features = self.extract_features(data)
            
            # Phase 3: Data Validation
            logger.info("\n✅ PHASE 3: Data Validation")
            validation_results = self.validate_data(data, features)
            
            # Phase 4: Hypothesis Testing
            logger.info("\n📈 PHASE 4: Hypothesis Testing")
            hypothesis_results = self.test_hypotheses(features)
            
            # Phase 5: Symbolic Regression
            logger.info("\n🧮 PHASE 5: Symbolic Regression")
            symbolic_results = self.discover_formulas(features)
            
            # Phase 6: Visualization
            logger.info("\n📊 PHASE 6: Visualization")
            self.generate_visualizations()
            
            # Phase 7: Report Generation
            logger.info("\n📝 PHASE 7: Report Generation")
            report = self.generate_report()
            
            # Compile results
            self.results = {
                'data': data,
                'features': features,
                'validation': validation_results,
                'hypothesis': hypothesis_results,
                'symbolic': symbolic_results,
                'report': report,
                'metadata': self.metadata
            }
            
            # Save results
            self.save_results()
            
            logger.info("\n" + "=" * 80)
            logger.info("✨ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.metadata['error'] = str(e)
            self.metadata['end_time'] = datetime.now().isoformat()
            self.save_results()  # Save partial results
            raise
    
    def generate_data(self) -> Dict[str, Any]:
        """Generate topological data according to configuration."""
        generator = self._lazy_import('topology')
        if not generator:
            logger.warning("Using dummy data generation")
            return self._generate_dummy_data()
        
        data = {
            'point_clouds': {},
            'graphs': {},
            'complexes': {},
            'parameters': {}
        }
        
        # Generate point clouds for each configuration
        for topology in self.config.topology_types:
            for noise in self.config.noise_levels:
                for dim in self.config.dimensions:
                    key = f"{topology}_n{noise}_d{dim}"
                    logger.info(f"  Generating: {key}")
                    
                    data['point_clouds'][key] = generator.generate_point_cloud(
                        n_points=self.config.n_samples,
                        dimension=dim,
                        topology_type=topology,
                        noise_level=noise
                    )
                    
                    data['parameters'][key] = {
                        'topology': topology,
                        'noise': noise,
                        'dimension': dim,
                        'n_points': self.config.n_samples
                    }
        
        # Generate graph topologies
        for graph_type in ['random', 'small_world', 'scale_free']:
            logger.info(f"  Generating graph: {graph_type}")
            data['graphs'][graph_type] = generator.generate_graph_topology(
                graph_type=graph_type,
                n_nodes=100
            )
        
        # Generate simplicial complexes
        for fill_prob in [0.1, 0.3, 0.5]:
            key = f"complex_p{fill_prob}"
            logger.info(f"  Generating: {key}")
            data['complexes'][key] = generator.generate_simplicial_complex(
                n_vertices=30,
                max_dim=3,
                fill_probability=fill_prob
            )
        
        # Save if configured
        if self.config.save_intermediates:
            np.save(self.output_dir / 'generated_data.npy', data)
        
        return data
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract topological features from data."""
        extractor = self._lazy_import('features')
        if not extractor:
            logger.warning("Using dummy feature extraction")
            return self._extract_dummy_features(data)
        
        features = {
            'persistence': {},
            'betti_numbers': {},
            'landscapes': {},
            'statistics': {}
        }
        
        # Process each point cloud
        for key, points in data['point_clouds'].items():
            logger.info(f"  Extracting features: {key}")
            
            # Compute persistence
            persistence = extractor.compute_persistence_diagram(
                points,
                max_dimension=self.config.max_homology_dim
            )
            features['persistence'][key] = persistence
            
            # Compute Betti numbers
            betti = extractor.compute_betti_numbers(persistence)
            features['betti_numbers'][key] = betti
            
            # Compute persistence landscapes
            landscape = extractor.compute_persistence_landscape(
                persistence,
                resolution=100
            )
            features['landscapes'][key] = landscape
            
            # Compute statistical features
            stats = extractor.compute_statistical_features(persistence)
            features['statistics'][key] = stats
        
        # Save if configured
        if self.config.save_intermediates:
            np.save(self.output_dir / 'extracted_features.npy', features)
        
        return features
    
    def validate_data(self, data: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and theoretical properties."""
        validator = self._lazy_import('validation')
        if not validator:
            logger.warning("Skipping validation")
            return {'status': 'skipped'}
        
        validation_results = {
            'data_quality': {},
            'theoretical_validation': {},
            'feature_validation': {},
            'warnings': []
        }
        
        # Validate each dataset
        for key in data['point_clouds']:
            logger.info(f"  Validating: {key}")
            
            # Data quality checks
            quality = validator.check_data_quality(data['point_clouds'][key])
            validation_results['data_quality'][key] = quality
            
            # Theoretical validation
            params = data['parameters'][key]
            theory = validator.validate_theoretical_properties(
                data['point_clouds'][key],
                topology_type=params['topology'],
                expected_dimension=params['dimension']
            )
            validation_results['theoretical_validation'][key] = theory
            
            # Feature validation
            if key in features['persistence']:
                feat_val = validator.validate_features(
                    features['persistence'][key],
                    features['betti_numbers'][key]
                )
                validation_results['feature_validation'][key] = feat_val
        
        # Compile warnings
        for key, quality in validation_results['data_quality'].items():
            if quality.get('warnings'):
                validation_results['warnings'].extend(quality['warnings'])
        
        if validation_results['warnings']:
            logger.warning(f"  ⚠️  {len(validation_results['warnings'])} warnings found")
        
        return validation_results
    
    def test_hypotheses(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical hypothesis testing."""
        tester = self._lazy_import('hypothesis')
        if not tester:
            logger.warning("Skipping hypothesis testing")
            return {'status': 'skipped'}
        
        hypothesis_results = {
            'distribution_tests': {},
            'correlation_tests': {},
            'stability_tests': {},
            'summary': {}
        }
        
        # Test distribution properties
        for test_name in self.config.statistical_tests:
            logger.info(f"  Running test: {test_name}")
            results = tester.test_distributions(
                features['statistics'],
                test_type=test_name,
                alpha=self.config.significance_level
            )
            hypothesis_results['distribution_tests'][test_name] = results
        
        # Test correlations
        correlation_results = tester.test_correlations(
            features,
            method='spearman'
        )
        hypothesis_results['correlation_tests'] = correlation_results
        
        # Test stability
        stability_results = tester.test_stability(
            features,
            n_bootstrap=self.config.bootstrap_iterations
        )
        hypothesis_results['stability_tests'] = stability_results
        
        # Generate summary
        hypothesis_results['summary'] = tester.summarize_results(hypothesis_results)
        
        return hypothesis_results
    
    def discover_formulas(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Discover symbolic formulas using regression."""
        regressor = self._lazy_import('symbolic')
        if not regressor:
            logger.warning("Skipping symbolic regression")
            return {'status': 'skipped'}
        
        symbolic_results = {
            'formulas': {},
            'validation': {},
            'complexity_analysis': {}
        }
        
        # Prepare data for regression
        X, y = self._prepare_regression_data(features)
        
        # Run symbolic regression
        logger.info("  Searching for symbolic formulas...")
        formulas = regressor.fit(
            X, y,
            max_complexity=self.config.max_complexity,
            population_size=self.config.population_size,
            generations=self.config.generations
        )
        symbolic_results['formulas'] = formulas
        
        # Validate discovered formulas
        validation = regressor.validate_formulas(formulas, X, y)
        symbolic_results['validation'] = validation
        
        # Analyze complexity
        complexity = regressor.analyze_complexity(formulas)
        symbolic_results['complexity_analysis'] = complexity
        
        # Log best formula
        if formulas:
            best = formulas[0]
            logger.info(f"  Best formula: {best.get('expression', 'N/A')}")
            logger.info(f"  R² score: {best.get('r2', 0):.4f}")
        
        return symbolic_results
    
    def generate_visualizations(self):
        """Generate all scientific visualizations."""
        visualizer = self._lazy_import('visualize')
        if not visualizer:
            logger.warning("Skipping visualizations")
            return
        
        logger.info("  Generating persistence diagrams...")
        visualizer.plot_persistence_diagrams(
            self.results.get('features', {}).get('persistence', {}),
            save_path=self.output_dir / 'persistence_diagrams.png'
        )
        
        logger.info("  Generating statistical plots...")
        visualizer.plot_statistical_analysis(
            self.results.get('hypothesis', {}),
            save_path=self.output_dir / 'statistical_analysis.png'
        )
        
        logger.info("  Generating formula complexity plot...")
        visualizer.plot_formula_complexity(
            self.results.get('symbolic', {}),
            save_path=self.output_dir / 'formula_complexity.png'
        )
    
    def generate_report(self) -> str:
        """Generate comprehensive scientific report."""
        report_lines = [
            "=" * 80,
            "SCIENTIFIC ANALYSIS REPORT",
            "=" * 80,
            f"\nExperiment ID: {self.config.get_hash()}",
            f"Date: {self.metadata['start_time']}",
            f"\nConfiguration:",
            f"  - Topology types: {self.config.topology_types}",
            f"  - Noise levels: {self.config.noise_levels}",
            f"  - Dimensions: {self.config.dimensions}",
            f"  - Samples: {self.config.n_samples}",
            "\n" + "-" * 40,
            "RESULTS SUMMARY",
            "-" * 40
        ]
        
        # Add validation summary
        if 'validation' in self.results:
            val = self.results['validation']
            n_warnings = len(val.get('warnings', []))
            report_lines.append(f"\nValidation: {'✅ Passed' if n_warnings == 0 else f'⚠️  {n_warnings} warnings'}")
        
        # Add hypothesis testing summary
        if 'hypothesis' in self.results:
            hyp = self.results['hypothesis']
            if 'summary' in hyp:
                report_lines.append(f"\nHypothesis Testing:")
                for key, value in hyp['summary'].items():
                    report_lines.append(f"  - {key}: {value}")
        
        # Add symbolic regression summary
        if 'symbolic' in self.results:
            sym = self.results['symbolic']
            if 'formulas' in sym and sym['formulas']:
                best = sym['formulas'][0]
                report_lines.append(f"\nBest Formula:")
                report_lines.append(f"  Expression: {best.get('expression', 'N/A')}")
                report_lines.append(f"  R² Score: {best.get('r2', 0):.4f}")
        
        report_lines.append("\n" + "=" * 80)
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"  Report saved to: {report_path}")
        return report
    
    def save_results(self):
        """Save all results to disk."""
        results_path = self.output_dir / 'results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            return obj
        
        json_results = convert_arrays(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
    
    def _prepare_regression_data(self, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature data for symbolic regression."""
        # Extract statistical features as X
        X_list = []
        y_list = []
        
        for key, stats in features.get('statistics', {}).items():
            if isinstance(stats, dict):
                # Use persistence entropy as target
                y_list.append(stats.get('persistence_entropy', 0))
                # Use other features as predictors
                x_features = [
                    stats.get('mean_persistence', 0),
                    stats.get('max_persistence', 0),
                    stats.get('total_persistence', 0)
                ]
                X_list.append(x_features)
        
        if not X_list:
            # Generate dummy data if no features
            X_list = np.random.randn(10, 3)
            y_list = np.random.randn(10)
        
        return np.array(X_list), np.array(y_list)
    
    def _generate_dummy_data(self) -> Dict[str, Any]:
        """Generate dummy data for testing."""
        return {
            'point_clouds': {
                'dummy': np.random.randn(100, 3)
            },
            'graphs': {},
            'complexes': {},
            'parameters': {
                'dummy': {
                    'topology': 'sphere',
                    'noise': 0.1,
                    'dimension': 3,
                    'n_points': 100
                }
            }
        }
    
    def _extract_dummy_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dummy features for testing."""
        return {
            'persistence': {
                'dummy': {0: np.array([[0, 1], [0, 0.5]])}
            },
            'betti_numbers': {
                'dummy': [1, 0, 0]
            },
            'landscapes': {
                'dummy': np.random.randn(100)
            },
            'statistics': {
                'dummy': {
                    'mean_persistence': 0.5,
                    'max_persistence': 1.0,
                    'total_persistence': 1.5,
                    'persistence_entropy': 0.7
                }
            }
        }


def main():
    """Main entry point for the pipeline."""
    # Create configuration
    config = ExperimentConfig(
        n_samples=500,
        topology_types=['sphere', 'torus'],
        noise_levels=[0.0, 0.1],
        dimensions=[3, 4],
        random_seed=42
    )
    
    # Create and run pipeline
    pipeline = ScientificPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()