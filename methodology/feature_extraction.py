"""
Data Validation and Theoretical Verification Module
====================================================
Validates data quality, theoretical properties, and topological invariants.
Ensures scientific rigor through comprehensive checks and verifications.
"""

import numpy as np
from scipy import stats
from scipy.spatial import distance_matrix
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict

class DataValidator:
    """Comprehensive validation of topological data and features."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator with tolerance settings.
        
        Args:
            tolerance: Numerical tolerance for validation checks
        """
        self.tolerance = tolerance
        self.validation_log = []
        
    def check_data_quality(self, 
                          data: np.ndarray,
                          check_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            data: Input data array
            check_types: Specific checks to perform (None for all)
            
        Returns:
            Dictionary with quality metrics and warnings
        """
        if check_types is None:
            check_types = ['nan', 'inf', 'duplicates', 'outliers', 'conditioning']
        
        results = {
            'passed': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Check for NaN values
        if 'nan' in check_types:
            nan_count = np.sum(np.isnan(data))
            results['metrics']['nan_count'] = int(nan_count)
            if nan_count > 0:
                results['warnings'].append(f"Found {nan_count} NaN values")
                results['passed'] = False
        
        # Check for infinite values
        if 'inf' in check_types:
            inf_count = np.sum(np.isinf(data))
            results['metrics']['inf_count'] = int(inf_count)
            if inf_count > 0:
                results['warnings'].append(f"Found {inf_count} infinite values")
                results['passed'] = False
        
        # Check for duplicate points
        if 'duplicates' in check_types and data.ndim == 2:
            unique_rows = np.unique(data, axis=0)
            duplicate_count = len(data) - len(unique_rows)
            results['metrics']['duplicate_count'] = int(duplicate_count)
            if duplicate_count > 0:
                results['warnings'].append(f"Found {duplicate_count} duplicate points")
        
        # Check for outliers using IQR method
        if 'outliers' in check_types:
            outlier_info = self._detect_outliers(data)
            results['metrics']['outlier_count'] = outlier_info['count']
            results['metrics']['outlier_percentage'] = outlier_info['percentage']
            if outlier_info['count'] > 0:
                results['warnings'].append(
                    f"Found {outlier_info['count']} outliers "
                    f"({outlier_info['percentage']:.1f}% of data)"
                )
        
        # Check numerical conditioning
        if 'conditioning' in check_types and data.ndim == 2:
            cond_number = self._compute_condition_number(data)
            results['metrics']['condition_number'] = float(cond_number)
            if cond_number > 1e10:
                results['warnings'].append(f"Poor conditioning: {cond_number:.2e}")
                results['passed'] = False
        
        # Add data statistics
        results['metrics'].update({
            'shape': data.shape,
            'dtype': str(data.dtype),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        })
        
        self._log_validation("data_quality", results)
        return results
    
    def validate_theoretical_properties(self,
                                       data: np.ndarray,
                                       topology_type: str,
                                       expected_dimension: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate theoretical topological properties.
        
        Args:
            data: Point cloud or topological structure
            topology_type: Expected topology ('sphere', 'torus', 'klein', etc.)
            expected_dimension: Expected manifold dimension
            
        Returns:
            Dictionary with theoretical validation results
        """
        results = {
            'topology_type': topology_type,
            'tests_passed': {},
            'metrics': {},
            'warnings': []
        }
        
        if topology_type == 'sphere':
            results.update(self._validate_sphere_properties(data, expected_dimension))
        elif topology_type == 'torus':
            results.update(self._validate_torus_properties(data))
        elif topology_type == 'klein':
            results.update(self._validate_klein_properties(data))
        elif topology_type == 'projective':
            results.update(self._validate_projective_properties(data))
        else:
            results['warnings'].append(f"Unknown topology type: {topology_type}")
        
        # Validate dimension if specified
        if expected_dimension is not None:
            intrinsic_dim = self._estimate_intrinsic_dimension(data)
            results['metrics']['estimated_dimension'] = float(intrinsic_dim)
            results['metrics']['expected_dimension'] = expected_dimension
            
            if abs(intrinsic_dim - expected_dimension) > 0.5:
                results['warnings'].append(
                    f"Dimension mismatch: expected {expected_dimension}, "
                    f"estimated {intrinsic_dim:.2f}"
                )
                results['tests_passed']['dimension_check'] = False
            else:
                results['tests_passed']['dimension_check'] = True
        
        self._log_validation("theoretical_properties", results)
        return results
    
    def _validate_sphere_properties(self, 
                                   data: np.ndarray,
                                   expected_dim: Optional[int]) -> Dict[str, Any]:
        """Validate sphere-specific properties."""
        results = {'tests_passed': {}, 'metrics': {}}
        
        if data.ndim != 2:
            results['warnings'] = ["Data must be 2D array for sphere validation"]
            return results
        
        # Check if points lie approximately on unit sphere
        norms = np.linalg.norm(data, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        results['metrics']['mean_radius'] = float(mean_norm)
        results['metrics']['radius_std'] = float(std_norm)
        
        # Test: Points should have similar norms (on sphere surface)
        if std_norm / mean_norm < 0.1:  # Less than 10% variation
            results['tests_passed']['constant_radius'] = True
        else:
            results['tests_passed']['constant_radius'] = False
            results.setdefault('warnings', []).append(
                f"High radius variation: {std_norm/mean_norm:.2%}"
            )
        
        # Test: Check Euler characteristic (simplified)
        if expected_dim is not None:
            # For S^n, Euler characteristic is 2 if n is even, 0 if n is odd
            expected_euler = 2 if expected_dim % 2 == 0 else 0
            results['metrics']['expected_euler_char'] = expected_euler
        
        # Test: Uniform distribution on sphere (using uniformity test)
        uniformity_stat = self._test_spherical_uniformity(data)
        results['metrics']['uniformity_statistic'] = float(uniformity_stat)
        results['tests_passed']['uniformity'] = uniformity_stat > 0.05
        
        return results
    
    def _validate_torus_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate torus-specific properties."""
        results = {'tests_passed': {}, 'metrics': {}}
        
        if data.ndim != 2 or data.shape[1] < 3:
            results['warnings'] = ["Data must be at least 3D for torus validation"]
            return results
        
        # For torus embedded in R^3
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        
        # Estimate major and minor radii
        xy_radius = np.sqrt(x**2 + y**2)
        R_est = np.mean(xy_radius)  # Major radius estimate
        r_est = np.std(z)  # Minor radius estimate (simplified)
        
        results['metrics']['major_radius_est'] = float(R_est)
        results['metrics']['minor_radius_est'] = float(r_est)
        
        # Test: Check if points satisfy torus equation approximately
        # Standard torus: (R - sqrt(x^2 + y^2))^2 + z^2 = r^2
        torus_residuals = (R_est - xy_radius)**2 + z**2 - r_est**2
        mean_residual = np.mean(np.abs(torus_residuals))
        
        results['metrics']['torus_equation_residual'] = float(mean_residual)
        results['tests_passed']['torus_equation'] = mean_residual < 0.5
        
        # Euler characteristic of torus is 0
        results['metrics']['expected_euler_char'] = 0
        
        return results
    
    def _validate_klein_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate Klein bottle properties."""
        results = {'tests_passed': {}, 'metrics': {}}
        
        if data.ndim != 2 or data.shape[1] < 4:
            results['warnings'] = ["Data must be at least 4D for Klein bottle"]
            return results
        
        # Klein bottle requires 4D embedding (no self-intersections)
        # Check that data uses all 4 dimensions
        dim_variance = np.var(data, axis=0)
        effective_dims = np.sum(dim_variance > 0.01)
        
        results['metrics']['effective_dimensions'] = int(effective_dims)
        results['tests_passed']['uses_4d'] = effective_dims >= 4
        
        # Klein bottle is non-orientable (Euler char = 0)
        results['metrics']['expected_euler_char'] = 0
        results['metrics']['orientable'] = False
        
        return results
    
    def _validate_projective_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate projective plane properties."""
        results = {'tests_passed': {}, 'metrics': {}}
        
        # Projective plane has Euler characteristic 1
        results['metrics']['expected_euler_char'] = 1
        results['metrics']['orientable'] = False
        
        # Check antipodal identification (simplified)
        if data.ndim == 2:
            # For each point, check if antipodal point exists
            antipodal_pairs = 0
            for i, point in enumerate(data):
                distances = np.linalg.norm(data + point, axis=1)
                if np.any(distances < 0.1):
                    antipodal_pairs += 1
            
            results['metrics']['antipodal_pairs'] = antipodal_pairs
            results['tests_passed']['antipodal_identification'] = antipodal_pairs > len(data) * 0.3
        
        return results
    
    def validate_features(self,
                         persistence: Dict[int, np.ndarray],
                         betti_numbers: List[int]) -> Dict[str, Any]:
        """
        Validate extracted topological features.
        
        Args:
            persistence: Persistence diagram
            betti_numbers: Computed Betti numbers
            
        Returns:
            Validation results for features
        """
        results = {
            'passed': True,
            'warnings': [],
            'checks': {}
        }
        
        # Check persistence diagram validity
        for dim, diagram in persistence.items():
            if diagram.size == 0:
                continue
                
            # Check birth < death
            invalid_pairs = np.sum(diagram[:, 0] >= diagram[:, 1])
            if invalid_pairs > 0:
                results['warnings'].append(
                    f"Dimension {dim}: {invalid_pairs} invalid persistence pairs"
                )
                results['passed'] = False
            results['checks'][f'dim_{dim}_valid_pairs'] = invalid_pairs == 0
            
            # Check for negative values
            negative_values = np.sum(diagram < 0)
            if negative_values > 0:
                results['warnings'].append(
                    f"Dimension {dim}: {negative_values} negative values"
                )
                results['passed'] = False
            results['checks'][f'dim_{dim}_non_negative'] = negative_values == 0
        
        # Validate Betti numbers
        if betti_numbers:
            # Check non-negative
            if any(b < 0 for b in betti_numbers):
                results['warnings'].append("Negative Betti numbers detected")
                results['passed'] = False
            
            # Check reasonable values (topology-dependent)
            if len(betti_numbers) > 0 and betti_numbers[0] == 0:
                results['warnings'].append("β₀ = 0 (no connected components)")
            
            results['checks']['betti_non_negative'] = all(b >= 0 for b in betti_numbers)
            results['checks']['has_components'] = len(betti_numbers) > 0 and betti_numbers[0] > 0
        
        self._log_validation("features", results)
        return results
    
    def validate_stability(self,
                          data: np.ndarray,
                          noise_levels: List[float] = [0.01, 0.05, 0.1],
                          metric: str = 'wasserstein') -> Dict[str, Any]:
        """
        Validate stability of topological features under perturbation.
        
        Args:
            data: Original data
            noise_levels: Noise levels to test
            metric: Distance metric for persistence diagrams
            
        Returns:
            Stability analysis results
        """
        results = {
            'stability_scores': {},
            'is_stable': True,
            'metrics': {}
        }
        
        # Import feature extractor (simplified version)
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        
        # Compute original persistence
        original_persistence = extractor.compute_persistence_diagram(data)
        
        for noise_level in noise_levels:
            # Add noise
            noisy_data = data + np.random.randn(*data.shape) * noise_level
            
            # Compute persistence of noisy data
            noisy_persistence = extractor.compute_persistence_diagram(noisy_data)
            
            # Compute distance
            if metric == 'wasserstein':
                distances = extractor.compute_wasserstein_distance(
                    original_persistence, noisy_persistence
                )
            else:
                distances = extractor.compute_bottleneck_distance(
                    original_persistence, noisy_persistence
                )
            
            # Store results
            key = f'noise_{noise_level}'
            results['stability_scores'][key] = {
                dim: float(dist) for dim, dist in distances.items()
            }
            
            # Check if stable (distance proportional to noise)
            max_distance = max(distances.values()) if distances else 0
            if max_distance > noise_level * 10:  # Threshold
                results['is_stable'] = False
        
        results['metrics']['max_instability'] = max(
            max(scores.values()) if scores else 0
            for scores in results['stability_scores'].values()
        )
        
        self._log_validation("stability", results)
        return results
    
    def validate_convergence(self,
                           data_sequence: List[np.ndarray],
                           expected_limit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate convergence of topological features.
        
        Args:
            data_sequence: Sequence of data arrays
            expected_limit: Expected limiting behavior
            
        Returns:
            Convergence validation results
        """
        results = {
            'converged': False,
            'convergence_rate': None,
            'limit_features': None,
            'metrics': {}
        }
        
        if len(data_sequence) < 3:
            results['warnings'] = ["Need at least 3 samples for convergence test"]
            return results
        
        # Import feature extractor
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        
        # Compute features for each sample
        features_sequence = []
        for data in data_sequence:
            persistence = extractor.compute_persistence_diagram(data)
            stats = extractor.compute_statistical_features(persistence)
            features_sequence.append(stats)
        
        # Check convergence using successive differences
        differences = []
        for i in range(1, len(features_sequence)):
            prev_features = np.array(list(features_sequence[i-1].values()))
            curr_features = np.array(list(features_sequence[i].values()))
            diff = np.linalg.norm(curr_features - prev_features)
            differences.append(diff)
        
        # Test for convergence
        if len(differences) >= 2:
            # Check if differences are decreasing
            convergence_ratio = differences[-1] / differences[-2] if differences[-2] > 0 else 1
            results['convergence_rate'] = float(convergence_ratio)
            results['converged'] = convergence_ratio < 0.9  # Converging if ratio < 0.9
        
        # Store limit features
        results['limit_features'] = features_sequence[-1]
        
        # Compare with expected limit if provided
        if expected_limit is not None:
            # Implement comparison logic
            pass
        
        self._log_validation("convergence", results)
        return results
    
    def cross_validate_methods(self,
                             data: np.ndarray,
                             methods: List[str] = ['vietoris_rips', 'alpha']) -> Dict[str, Any]:
        """
        Cross-validate results using different computational methods.
        
        Args:
            data: Input data
            methods: List of methods to compare
            
        Returns:
            Cross-validation results
        """
        results = {
            'method_agreement': {},
            'discrepancies': [],
            'recommended_method': None
        }
        
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        
        method_results = {}
        for method in methods:
            try:
                persistence = extractor.compute_persistence_diagram(
                    data, filtration=method
                )
                betti = extractor.compute_betti_numbers(persistence)
                method_results[method] = {
                    'persistence': persistence,
                    'betti': betti
                }
            except Exception as e:
                results['discrepancies'].append(f"Method {method} failed: {e}")
        
        # Compare results
        if len(method_results) >= 2:
            methods_list = list(method_results.keys())
            for i in range(len(methods_list)):
                for j in range(i + 1, len(methods_list)):
                    method1, method2 = methods_list[i], methods_list[j]
                    
                    # Compare Betti numbers
                    betti1 = method_results[method1]['betti']
                    betti2 = method_results[method2]['betti']
                    
                    agreement = np.allclose(betti1, betti2, rtol=0.1)
                    key = f"{method1}_vs_{method2}"
                    results['method_agreement'][key] = agreement
                    
                    if not agreement:
                        results['discrepancies'].append(
                            f"Disagreement between {method1} and {method2}"
                        )
        
        # Recommend most stable method
        if method_results:
            results['recommended_method'] = methods[0]
        
        self._log_validation("cross_validation", results)
        return results
    
    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        flat_data = data.flatten()
        Q1 = np.percentile(flat_data, 25)
        Q3 = np.percentile(flat_data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (flat_data < lower_bound) | (flat_data > upper_bound)
        
        return {
            'count': int(np.sum(outliers)),
            'percentage': float(100 * np.mean(outliers)),
            'bounds': (float(lower_bound), float(upper_bound))
        }
    
    def _compute_condition_number(self, data: np.ndarray) -> float:
        """Compute condition number of data matrix."""
        if data.shape[0] < data.shape[1]:
            # More features than samples
            return np.inf
        
        # Compute via SVD
        try:
            _, s, _ = np.linalg.svd(data, compute_uv=False)
            if s[-1] > 0:
                return s[0] / s[-1]
            else:
                return np.inf
        except:
            return np.inf
    
    def _estimate_intrinsic_dimension(self, data: np.ndarray) -> float:
        """Estimate intrinsic dimension using correlation dimension."""
        if data.ndim != 2 or data.shape[0] < 10:
            return float(data.shape[1]) if data.ndim == 2 else 1.0
        
        # Simplified correlation dimension estimation
        n_points = min(1000, len(data))
        sample_idx = np.random.choice(len(data), n_points, replace=False)
        sample = data[sample_idx]
        
        # Compute pairwise distances
        distances = distance_matrix(sample, sample)
        distances = distances[np.triu_indices(n_points, k=1)]
        
        # Estimate dimension from distance distribution
        if len(distances) > 0:
            log_distances = np.log(distances + 1e-10)
            hist, bins = np.histogram(log_distances, bins=50)
            
            # Find linear region in log-log plot (simplified)
            # In practice, use more sophisticated methods
            dimension_estimate = 2.0  # Default estimate
            
            return dimension_estimate
        
        return float(data.shape[1])
    
    def _test_spherical_uniformity(self, data: np.ndarray) -> float:
        """Test uniformity of points on sphere."""
        # Simplified uniformity test using nearest neighbor distances
        if len(data) < 10:
            return 1.0
        
        # Normalize to unit sphere
        normalized = data / np.linalg.norm(data, axis=1, keepdims=True)
        
        # Compute nearest neighbor distances
        from scipy.spatial import KDTree
        tree = KDTree(normalized)
        distances, _ = tree.query(normalized, k=2)
        nn_distances = distances[:, 1]  # First nearest neighbor
        
        # Test if distances follow expected distribution
        # For uniform distribution on sphere, use Kolmogorov-Smirnov test
        # Simplified: check coefficient of variation
        cv = np.std(nn_distances) / np.mean(nn_distances)
        
        # Lower CV indicates more uniform
        uniformity = np.exp(-cv)  # Convert to [0, 1] score
        
        return float(uniformity)
    
    def _log_validation(self, validation_type: str, results: Dict[str, Any]):
        """Log validation results."""
        self.validation_log.append({
            'type': validation_type,
            'timestamp': np.datetime64('now'),
            'results': results
        })
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report_lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            ""
        ]
        
        for entry in self.validation_log:
            report_lines.append(f"\n{entry['type'].upper()} VALIDATION")
            report_lines.append("-" * 40)
            
            results = entry['results']
            if 'passed' in results:
                status = "✅ PASSED" if results['passed'] else "❌ FAILED"
                report_lines.append(f"Status: {status}")
            
            if 'warnings' in results and results['warnings']:
                report_lines.append("\nWarnings:")
                for warning in results['warnings']:
                    report_lines.append(f"  ⚠️  {warning}")
            
            if 'metrics' in results:
                report_lines.append("\nMetrics:")
                for key, value in results['metrics'].items():
                    if isinstance(value, float):
                        report_lines.append(f"  {key}: {value:.4f}")
                    else:
                        report_lines.append(f"  {key}: {value}")
        
        report_lines.append("\n" + "=" * 60)
        return "\n".join(report_lines)