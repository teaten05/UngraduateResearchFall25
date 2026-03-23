"""
Statistical Hypothesis Testing Module
======================================
Comprehensive statistical tests for topological data analysis.
Implements distribution tests, correlation analysis, and stability testing.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp, anderson, chi2_contingency, spearmanr, pearsonr
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from collections import defaultdict
from itertools import combinations

class HypothesisTester:
    """Perform statistical hypothesis testing on topological features."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize hypothesis tester.
        
        Args:
            significance_level: Significance level for tests (alpha)
        """
        self.alpha = significance_level
        self.test_results = []
        
    def test_distributions(self,
                          features: Dict[str, Dict[str, float]],
                          test_type: str = 'ks',
                          reference_distribution: Optional[str] = None,
                          alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Test distributional properties of features.
        
        Args:
            features: Dictionary of feature sets
            test_type: Type of test ('ks', 'anderson', 'shapiro', 'chi2')
            reference_distribution: Reference distribution for comparison
            alpha: Significance level (uses default if None)
            
        Returns:
            Test results with p-values and decisions
        """
        if alpha is None:
            alpha = self.alpha
            
        results = {
            'test_type': test_type,
            'alpha': alpha,
            'tests': {},
            'summary': {}
        }
        
        # Convert features to arrays
        feature_arrays = {}
        for key, feat_dict in features.items():
            if isinstance(feat_dict, dict):
                feature_arrays[key] = np.array(list(feat_dict.values()))
            else:
                feature_arrays[key] = np.array(feat_dict)
        
        if test_type == 'ks':
            results['tests'] = self._kolmogorov_smirnov_tests(feature_arrays, alpha)
        elif test_type == 'anderson':
            results['tests'] = self._anderson_darling_tests(feature_arrays, reference_distribution)
        elif test_type == 'shapiro':
            results['tests'] = self._shapiro_wilk_tests(feature_arrays, alpha)
        elif test_type == 'chi2':
            results['tests'] = self._chi_square_tests(feature_arrays, alpha)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Summarize results
        n_rejected = sum(1 for test in results['tests'].values() 
                        if test.get('reject_null', False))
        n_total = len(results['tests'])
        
        results['summary'] = {
            'n_tests': n_total,
            'n_rejected': n_rejected,
            'rejection_rate': n_rejected / n_total if n_total > 0 else 0,
            'fdr_corrected': self._apply_fdr_correction(results['tests'], alpha)
        }
        
        self._log_test("distribution", results)
        return results
    
    def _kolmogorov_smirnov_tests(self,
                                 feature_arrays: Dict[str, np.ndarray],
                                 alpha: float) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov tests."""
        tests = {}
        
        # Test each pair of feature sets
        keys = list(feature_arrays.keys())
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys[i+1:], i+1):
                test_name = f"{key1}_vs_{key2}"
                
                # Two-sample KS test
                statistic, p_value = ks_2samp(
                    feature_arrays[key1],
                    feature_arrays[key2]
                )
                
                tests[test_name] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'reject_null': p_value < alpha,
                    'interpretation': 'Different distributions' if p_value < alpha else 'Same distribution'
                }
        
        # Test against theoretical distributions
        for key, data in feature_arrays.items():
            # Test for normality
            if len(data) > 0:
                # Standardize data
                standardized = (data - np.mean(data)) / (np.std(data) + 1e-10)
                
                # Test against standard normal
                statistic, p_value = stats.kstest(standardized, 'norm')
                
                tests[f"{key}_normality"] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'reject_null': p_value < alpha,
                    'interpretation': 'Not normal' if p_value < alpha else 'Possibly normal'
                }
        
        return tests
    
    def _anderson_darling_tests(self,
                               feature_arrays: Dict[str, np.ndarray],
                               dist_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform Anderson-Darling tests."""
        tests = {}
        
        if dist_name is None:
            dist_name = 'norm'
        
        for key, data in feature_arrays.items():
            if len(data) < 5:
                tests[key] = {
                    'statistic': np.nan,
                    'critical_values': [],
                    'significance_levels': [],
                    'interpretation': 'Insufficient data'
                }
                continue
            
            try:
                result = anderson(data, dist=dist_name)
                
                # Determine rejection at 5% level
                idx_5pct = 2  # Index for 5% significance level (typically)
                reject = result.statistic > result.critical_values[idx_5pct]
                
                tests[key] = {
                    'statistic': float(result.statistic),
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist(),
                    'reject_null': reject,
                    'interpretation': f"Not {dist_name}" if reject else f"Possibly {dist_name}"
                }
            except Exception as e:
                tests[key] = {
                    'error': str(e),
                    'interpretation': 'Test failed'
                }
        
        return tests
    
    def _shapiro_wilk_tests(self,
                           feature_arrays: Dict[str, np.ndarray],
                           alpha: float) -> Dict[str, Any]:
        """Perform Shapiro-Wilk normality tests."""
        tests = {}
        
        for key, data in feature_arrays.items():
            if len(data) < 3:
                tests[key] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'interpretation': 'Insufficient data'
                }
                continue
            
            if len(data) > 5000:
                # Shapiro-Wilk not recommended for large samples
                tests[key] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'interpretation': 'Sample too large for Shapiro-Wilk'
                }
                continue
            
            statistic, p_value = stats.shapiro(data)
            
            tests[key] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'reject_null': p_value < alpha,
                'interpretation': 'Not normal' if p_value < alpha else 'Possibly normal'
            }
        
        return tests
    
    def _chi_square_tests(self,
                         feature_arrays: Dict[str, np.ndarray],
                         alpha: float) -> Dict[str, Any]:
        """Perform chi-square tests."""
        tests = {}
        
        # Discretize continuous features for chi-square test
        discretized = {}
        n_bins = 10
        
        for key, data in feature_arrays.items():
            if len(data) > 0:
                # Discretize using equal-width bins
                discretized[key] = np.histogram(data, bins=n_bins)[0]
        
        # Test independence between pairs
        keys = list(discretized.keys())
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys[i+1:], i+1):
                test_name = f"{key1}_vs_{key2}_independence"
                
                # Create contingency table
                contingency = np.vstack([discretized[key1], discretized[key2]])
                
                # Chi-square test of independence
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                tests[test_name] = {
                    'statistic': float(chi2),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'reject_null': p_value < alpha,
                    'interpretation': 'Dependent' if p_value < alpha else 'Independent'
                }
        
        return tests
    
    def test_correlations(self,
                         features: Dict[str, Any],
                         method: str = 'spearman',
                         alpha: Optional[float] = None) -> Dict[str, Any]:
        """
        Test correlations between features.
        
        Args:
            features: Feature dictionary
            method: Correlation method ('spearman', 'pearson', 'kendall')
            alpha: Significance level
            
        Returns:
            Correlation test results
        """
        if alpha is None:
            alpha = self.alpha
            
        results = {
            'method': method,
            'correlations': {},
            'significant_correlations': [],
            'correlation_matrix': None
        }
        
        # Extract numerical features
        feature_vectors = []
        feature_names = []
        
        for category, features_dict in features.items():
            if isinstance(features_dict, dict):
                for key, values in features_dict.items():
                    if isinstance(values, dict):
                        # Nested features
                        for sub_key, value in values.items():
                            if isinstance(value, (int, float)):
                                feature_names.append(f"{category}_{key}_{sub_key}")
                                feature_vectors.append([value])
                    elif isinstance(values, (list, np.ndarray)):
                        # Array features
                        flat = np.array(values).flatten()
                        for i, val in enumerate(flat[:10]):  # Limit to first 10
                            feature_names.append(f"{category}_{key}_{i}")
                            feature_vectors.append([val])
        
        if not feature_vectors:
            results['error'] = "No numerical features found"
            return results
        
        # Create feature matrix
        feature_matrix = np.column_stack(feature_vectors)
        
        # Compute correlations
        n_features = feature_matrix.shape[1]
        correlation_matrix = np.zeros((n_features, n_features))
        p_value_matrix = np.ones((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if method == 'spearman':
                    corr, p_val = spearmanr(feature_matrix[:, i], feature_matrix[:, j])
                elif method == 'pearson':
                    corr, p_val = pearsonr(feature_matrix[:, i], feature_matrix[:, j])
                elif method == 'kendall':
                    corr, p_val = stats.kendalltau(feature_matrix[:, i], feature_matrix[:, j])
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
                p_value_matrix[i, j] = p_val
                p_value_matrix[j, i] = p_val
                
                # Store significant correlations
                if p_val < alpha and not np.isnan(corr):
                    results['significant_correlations'].append({
                        'feature1': feature_names[i] if i < len(feature_names) else f"feat_{i}",
                        'feature2': feature_names[j] if j < len(feature_names) else f"feat_{j}",
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'strength': self._interpret_correlation_strength(corr)
                    })
        
        # Fill diagonal
        np.fill_diagonal(correlation_matrix, 1.0)
        np.fill_diagonal(p_value_matrix, 0.0)
        
        results['correlation_matrix'] = correlation_matrix.tolist()
        results['p_value_matrix'] = p_value_matrix.tolist()
        
        # Summary statistics
        results['summary'] = {
            'n_features': n_features,
            'n_significant': len(results['significant_correlations']),
            'max_correlation': float(np.max(np.abs(correlation_matrix[np.triu_indices(n_features, k=1)]))),
            'mean_correlation': float(np.mean(np.abs(correlation_matrix[np.triu_indices(n_features, k=1)])))
        }
        
        self._log_test("correlation", results)
        return results
    
    def test_stability(self,
                      features: Dict[str, Any],
                      n_bootstrap: int = 1000,
                      statistic_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Test stability of features using bootstrap.
        
        Args:
            features: Feature dictionary
            n_bootstrap: Number of bootstrap iterations
            statistic_func: Function to compute statistic (default: mean)
            
        Returns:
            Stability test results
        """
        if statistic_func is None:
            statistic_func = np.mean
            
        results = {
            'n_bootstrap': n_bootstrap,
            'stability_scores': {},
            'confidence_intervals': {},
            'coefficient_of_variation': {}
        }
        
        # Process each feature type
        for feature_type, feature_data in features.items():
            if not isinstance(feature_data, dict):
                continue
                
            for key, values in feature_data.items():
                # Convert to array
                if isinstance(values, dict):
                    data = np.array(list(values.values()))
                elif isinstance(values, (list, np.ndarray)):
                    data = np.array(values).flatten()
                else:
                    continue
                
                if len(data) < 2:
                    continue
                
                # Bootstrap
                bootstrap_stats = []
                for _ in range(n_bootstrap):
                    # Resample with replacement
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_stats.append(statistic_func(bootstrap_sample))
                
                bootstrap_stats = np.array(bootstrap_stats)
                
                # Compute stability metrics
                feature_key = f"{feature_type}_{key}"
                
                # Confidence interval
                ci_lower = np.percentile(bootstrap_stats, 2.5)
                ci_upper = np.percentile(bootstrap_stats, 97.5)
                results['confidence_intervals'][feature_key] = [float(ci_lower), float(ci_upper)]
                
                # Coefficient of variation
                cv = np.std(bootstrap_stats) / (np.mean(bootstrap_stats) + 1e-10)
                results['coefficient_of_variation'][feature_key] = float(cv)
                
                # Stability score (inverse of CV, bounded)
                stability = 1.0 / (1.0 + cv)
                results['stability_scores'][feature_key] = float(stability)
        
        # Overall stability assessment
        if results['stability_scores']:
            mean_stability = np.mean(list(results['stability_scores'].values()))
            min_stability = np.min(list(results['stability_scores'].values()))
            
            results['summary'] = {
                'mean_stability': float(mean_stability),
                'min_stability': float(min_stability),
                'is_stable': min_stability > 0.5,  # Threshold for stability
                'interpretation': 'Stable' if min_stability > 0.5 else 'Unstable'
            }
        else:
            results['summary'] = {
                'mean_stability': 0.0,
                'min_stability': 0.0,
                'is_stable': False,
                'interpretation': 'No features to test'
            }
        
        self._log_test("stability", results)
        return results
    
    def test_homogeneity(self,
                        groups: Dict[str, np.ndarray],
                        test: str = 'levene') -> Dict[str, Any]:
        """
        Test homogeneity of variance across groups.
        
        Args:
            groups: Dictionary of group data
            test: Test type ('levene', 'bartlett', 'fligner')
            
        Returns:
            Homogeneity test results
        """
        results = {
            'test': test,
            'groups': list(groups.keys()),
            'test_results': {}
        }
        
        # Prepare data
        group_arrays = [np.array(data).flatten() for data in groups.values()]
        
        if len(group_arrays) < 2:
            results['error'] = "Need at least 2 groups for homogeneity test"
            return results
        
        # Perform test
        if test == 'levene':
            statistic, p_value = stats.levene(*group_arrays)
        elif test == 'bartlett':
            statistic, p_value = stats.bartlett(*group_arrays)
        elif test == 'fligner':
            statistic, p_value = stats.fligner(*group_arrays)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        results['test_results'] = {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'reject_null': p_value < self.alpha,
            'interpretation': 'Heterogeneous variances' if p_value < self.alpha else 'Homogeneous variances'
        }
        
        # Compute variance ratios
        variances = [np.var(data) for data in group_arrays]
        if min(variances) > 0:
            max_ratio = max(variances) / min(variances)
            results['variance_ratio'] = float(max_ratio)
            results['variances'] = {
                key: float(np.var(groups[key]))
                for key in groups.keys()
            }
        
        self._log_test("homogeneity", results)
        return results
    
    def test_independence(self,
                         x: np.ndarray,
                         y: np.ndarray,
                         method: str = 'mutual_info') -> Dict[str, Any]:
        """
        Test independence between variables.
        
        Args:
            x: First variable
            y: Second variable
            method: Test method ('mutual_info', 'distance_correlation', 'hsic')
            
        Returns:
            Independence test results
        """
        results = {
            'method': method,
            'test_statistic': None,
            'p_value': None,
            'independent': None
        }
        
        if method == 'mutual_info':
            # Mutual information test
            from sklearn.feature_selection import mutual_info_regression
            
            # Reshape for sklearn
            x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x
            
            mi = mutual_info_regression(x_reshaped, y, random_state=42)
            results['test_statistic'] = float(np.mean(mi))
            
            # Bootstrap for p-value
            n_permutations = 1000
            null_distribution = []
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                mi_null = mutual_info_regression(x_reshaped, y_permuted, random_state=42)
                null_distribution.append(np.mean(mi_null))
            
            p_value = np.mean(null_distribution >= results['test_statistic'])
            results['p_value'] = float(p_value)
            
        elif method == 'distance_correlation':
            # Distance correlation test
            dcor = self._distance_correlation(x, y)
            results['test_statistic'] = float(dcor)
            
            # Permutation test for p-value
            n_permutations = 1000
            null_distribution = []
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                dcor_null = self._distance_correlation(x, y_permuted)
                null_distribution.append(dcor_null)
            
            p_value = np.mean(null_distribution >= dcor)
            results['p_value'] = float(p_value)
            
        elif method == 'hsic':
            # Hilbert-Schmidt Independence Criterion
            hsic = self._hsic(x, y)
            results['test_statistic'] = float(hsic)
            
            # Permutation test
            n_permutations = 1000
            null_distribution = []
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                hsic_null = self._hsic(x, y_permuted)
                null_distribution.append(hsic_null)
            
            p_value = np.mean(null_distribution >= hsic)
            results['p_value'] = float(p_value)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results['independent'] = results['p_value'] > self.alpha if results['p_value'] is not None else None
        results['interpretation'] = 'Independent' if results['independent'] else 'Dependent'
        
        self._log_test("independence", results)
        return results
    
    def multiple_testing_correction(self,
                                  p_values: List[float],
                                  method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple testing correction.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'fdr', 'holm')
            
        Returns:
            Corrected p-values and decisions
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        results = {
            'method': method,
            'n_tests': n_tests,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': None,
            'reject': None
        }
        
        if method == 'bonferroni':
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)
            
        elif method == 'fdr' or method == 'bh':
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            corrected_p = np.zeros_like(p_values)
            for i in range(n_tests):
                corrected_p[sorted_idx[i]] = sorted_p[i] * n_tests / (i + 1)
            
            # Ensure monotonicity
            for i in range(n_tests - 2, -1, -1):
                corrected_p[sorted_idx[i]] = min(corrected_p[sorted_idx[i]], 
                                                corrected_p[sorted_idx[i + 1]])
            
        elif method == 'holm':
            # Holm-Bonferroni
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            corrected_p = np.zeros_like(p_values)
            for i in range(n_tests):
                corrected_p[sorted_idx[i]] = sorted_p[i] * (n_tests - i)
            
            # Ensure monotonicity
            for i in range(1, n_tests):
                corrected_p[sorted_idx[i]] = max(corrected_p[sorted_idx[i]], 
                                                corrected_p[sorted_idx[i - 1]])
            
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        corrected_p = np.minimum(corrected_p, 1.0)
        results['corrected_p_values'] = corrected_p.tolist()
        results['reject'] = (corrected_p < self.alpha).tolist()
        results['n_rejected'] = int(np.sum(corrected_p < self.alpha))
        
        return results
    
    def summarize_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize all hypothesis testing results.
        
        Args:
            all_results: Dictionary of all test results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_tests': 0,
            'significant_results': 0,
            'test_types': [],
            'key_findings': [],
            'recommendations': []
        }
        
        # Count tests and significant results
        for category, results in all_results.items():
            if 'tests' in results:
                n_tests = len(results['tests'])
                n_significant = sum(1 for test in results['tests'].values()
                                  if test.get('reject_null', False))
                
                summary['total_tests'] += n_tests
                summary['significant_results'] += n_significant
                summary['test_types'].append(category)
                
                if n_significant > 0:
                    summary['key_findings'].append(
                        f"{category}: {n_significant}/{n_tests} significant"
                    )
        
        # Add stability results
        if 'stability_tests' in all_results:
            stability = all_results['stability_tests']
            if 'summary' in stability:
                summary['key_findings'].append(
                    f"Stability: {stability['summary']['interpretation']}"
                )
        
        # Add correlation results
        if 'correlation_tests' in all_results:
            correlations = all_results['correlation_tests']
            if 'significant_correlations' in correlations:
                n_sig_corr = len(correlations['significant_correlations'])
                if n_sig_corr > 0:
                    summary['key_findings'].append(
                        f"Found {n_sig_corr} significant correlations"
                    )
        
        # Generate recommendations
        if summary['significant_results'] > summary['total_tests'] * 0.5:
            summary['recommendations'].append(
                "High rate of significant results - consider multiple testing correction"
            )
        
        if 'stability_tests' in all_results:
            if not all_results['stability_tests'].get('summary', {}).get('is_stable', True):
                summary['recommendations'].append(
                    "Features show instability - increase sample size or regularization"
                )
        
        return summary
    
    def _distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance correlation between x and y."""
        n = len(x)
        
        # Compute distance matrices
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        a = squareform(pdist(x))
        b = squareform(pdist(y))
        
        # Double center the distance matrices
        a_centered = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        b_centered = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
        
        # Compute distance covariance
        dcov_xy = np.sqrt(np.mean(a_centered * b_centered))
        dcov_xx = np.sqrt(np.mean(a_centered * a_centered))
        dcov_yy = np.sqrt(np.mean(b_centered * b_centered))
        
        # Distance correlation
        if dcov_xx * dcov_yy > 0:
            dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            dcor = 0.0
        
        return dcor
    
    def _hsic(self, x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute Hilbert-Schmidt Independence Criterion."""
        n = len(x)
        
        # Reshape if needed
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Compute RBF kernels
        k_x = np.exp(-gamma * squareform(pdist(x, 'sqeuclidean')))
        k_y = np.exp(-gamma * squareform(pdist(y, 'sqeuclidean')))
        
        # Center the kernel matrices
        h = np.eye(n) - np.ones((n, n)) / n
        k_x_centered = h @ k_x @ h
        k_y_centered = h @ k_y @ h
        
        # Compute HSIC
        hsic = np.trace(k_x_centered @ k_y_centered) / (n - 1) ** 2
        
        return hsic
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "negligible"
        elif abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.5:
            return "moderate"
        elif abs_corr < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def _apply_fdr_correction(self, tests: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """Apply FDR correction to test results."""
        # Extract p-values
        p_values = []
        test_names = []
        
        for name, test in tests.items():
            if 'p_value' in test and not np.isnan(test['p_value']):
                p_values.append(test['p_value'])
                test_names.append(name)
        
        if not p_values:
            return {'n_corrected': 0, 'n_significant': 0}
        
        # Apply FDR correction
        corrected = self.multiple_testing_correction(p_values, method='fdr')
        
        return {
            'n_corrected': len(p_values),
            'n_significant': corrected['n_rejected'],
            'significant_tests': [
                test_names[i] for i, reject in enumerate(corrected['reject']) if reject
            ]
        }
    
    def _log_test(self, test_type: str, results: Dict[str, Any]):
        """Log test results."""
        self.test_results.append({
            'type': test_type,
            'timestamp': np.datetime64('now'),
            'results': results
        })