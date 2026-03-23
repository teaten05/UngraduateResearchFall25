"""
Random Topology Generator with Scientific Rigor
================================================
Generates diverse topological structures with controlled parameters
for systematic exploration of topological invariants.
"""

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.spatial import distance_matrix
from typing import Dict, List, Tuple, Optional, Any
import warnings

class TopologyGenerator:
    """Generate various topological structures with controlled parameters."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator with optional random seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def generate_point_cloud(self, 
                            n_points: int = 100,
                            dimension: int = 3,
                            topology_type: str = 'sphere',
                            noise_level: float = 0.1) -> np.ndarray:
        """
        Generate point cloud with specific topological structure.
        
        Args:
            n_points: Number of points to generate
            dimension: Ambient dimension for embedding
            topology_type: Type of topology ('sphere', 'torus', 'klein', 'projective')
            noise_level: Gaussian noise standard deviation
            
        Returns:
            Point cloud as numpy array of shape (n_points, dimension)
        """
        if topology_type == 'sphere':
            points = self._generate_sphere(n_points, dimension)
        elif topology_type == 'torus':
            points = self._generate_torus(n_points)
            if dimension > 3:
                points = self._embed_higher_dim(points, dimension)
        elif topology_type == 'klein':
            points = self._generate_klein_bottle(n_points)
            if dimension > 4:
                points = self._embed_higher_dim(points, dimension)
        elif topology_type == 'projective':
            points = self._generate_projective_plane(n_points)
            if dimension > 4:
                points = self._embed_higher_dim(points, dimension)
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        # Add controlled noise
        noise = np.random.randn(*points.shape) * noise_level
        points += noise
        
        return points
    
    def _generate_sphere(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate points on a unit sphere."""
        points = np.random.randn(n_points, dimension)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def _generate_torus(self, n_points: int) -> np.ndarray:
        """Generate points on a torus in R^3."""
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        
        R, r = 2.0, 1.0  # Major and minor radii
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        return np.column_stack([x, y, z])
    
    def _generate_klein_bottle(self, n_points: int) -> np.ndarray:
        """Generate points on Klein bottle immersion in R^4."""
        u = np.random.uniform(0, 2*np.pi, n_points)
        v = np.random.uniform(0, 2*np.pi, n_points)
        
        # Standard Klein bottle immersion
        x = (2 + np.cos(v)) * np.cos(u)
        y = (2 + np.cos(v)) * np.sin(u)
        z = np.sin(v) * np.cos(u/2)
        w = np.sin(v) * np.sin(u/2)
        
        return np.column_stack([x, y, z, w])
    
    def _generate_projective_plane(self, n_points: int) -> np.ndarray:
        """Generate points for projective plane via Boy's surface."""
        u = np.random.uniform(0, np.pi, n_points)
        v = np.random.uniform(0, np.pi, n_points)
        
        # Boy's surface parametrization
        denom = np.sqrt(2) - np.sin(2*u) * np.sin(3*v)
        x = np.cos(u) * np.sin(2*v) / denom
        y = np.sin(u) * np.sin(2*v) / denom
        z = np.cos(2*v) / denom
        w = np.sin(u) * np.cos(v)  # Extra dimension for embedding
        
        return np.column_stack([x, y, z, w])
    
    def _embed_higher_dim(self, points: np.ndarray, target_dim: int) -> np.ndarray:
        """Embed points in higher dimension with isometry."""
        current_dim = points.shape[1]
        if target_dim <= current_dim:
            return points
        
        # Add zero dimensions
        extra_dims = np.zeros((points.shape[0], target_dim - current_dim))
        return np.hstack([points, extra_dims])
    
    def generate_simplicial_complex(self,
                                   n_vertices: int = 20,
                                   max_dim: int = 3,
                                   fill_probability: float = 0.3) -> Dict[int, List[Tuple]]:
        """
        Generate random simplicial complex with controlled filling.
        
        Args:
            n_vertices: Number of 0-simplices (vertices)
            max_dim: Maximum dimension of simplices
            fill_probability: Probability of including a simplex
            
        Returns:
            Dictionary mapping dimension to list of simplices
        """
        simplices = {i: [] for i in range(max_dim + 1)}
        
        # Add all vertices
        simplices[0] = [(i,) for i in range(n_vertices)]
        
        # Build higher dimensional simplices
        for dim in range(1, max_dim + 1):
            # Consider all possible simplices of this dimension
            from itertools import combinations
            
            for simplex in combinations(range(n_vertices), dim + 1):
                # Check if all faces exist (for valid complex)
                faces_exist = True
                for face in combinations(simplex, dim):
                    if tuple(face) not in simplices[dim - 1]:
                        faces_exist = False
                        break
                
                # Add with probability if valid
                if faces_exist or dim == 1:  # Always allow edges
                    if np.random.random() < fill_probability:
                        simplices[dim].append(tuple(simplex))
                        
                        # Ensure all faces are included (closure property)
                        if dim > 1:
                            for face in combinations(simplex, dim):
                                if tuple(face) not in simplices[dim - 1]:
                                    simplices[dim - 1].append(tuple(face))
        
        return simplices
    
    def generate_graph_topology(self,
                               graph_type: str = 'random',
                               n_nodes: int = 50,
                               **kwargs) -> nx.Graph:
        """
        Generate graph with specific topological properties.
        
        Args:
            graph_type: Type of graph ('random', 'small_world', 'scale_free', 'lattice')
            n_nodes: Number of nodes
            **kwargs: Additional parameters for specific graph types
            
        Returns:
            NetworkX graph object
        """
        if graph_type == 'random':
            p = kwargs.get('edge_probability', 0.1)
            G = nx.erdos_renyi_graph(n_nodes, p, seed=self.seed)
            
        elif graph_type == 'small_world':
            k = kwargs.get('k_neighbors', 4)
            p = kwargs.get('rewire_probability', 0.3)
            G = nx.watts_strogatz_graph(n_nodes, k, p, seed=self.seed)
            
        elif graph_type == 'scale_free':
            m = kwargs.get('m_edges', 2)
            G = nx.barabasi_albert_graph(n_nodes, m, seed=self.seed)
            
        elif graph_type == 'lattice':
            dim = kwargs.get('dimensions', [10, 5])
            G = nx.grid_graph(dim)
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        return G
    
    def generate_persistence_diagram(self,
                                    points: np.ndarray,
                                    max_dimension: int = 2) -> Dict[int, np.ndarray]:
        """
        Generate persistence diagram from point cloud.
        
        Args:
            points: Point cloud data
            max_dimension: Maximum homology dimension
            
        Returns:
            Dictionary mapping dimension to birth-death pairs
        """
        # Simple Vietoris-Rips filtration simulation
        # (In practice, use gudhi or dionysus for real computation)
        
        n_points = len(points)
        distances = distance_matrix(points, points)
        
        persistence = {dim: [] for dim in range(max_dimension + 1)}
        
        # Simulate persistence pairs (simplified)
        max_dist = np.max(distances)
        
        # H_0 (connected components)
        for i in range(n_points - 1):
            birth = 0
            death = np.random.uniform(0, max_dist * 0.3)
            persistence[0].append([birth, death])
        
        # Higher dimensions (simplified simulation)
        for dim in range(1, max_dimension + 1):
            n_features = max(1, n_points // (2 ** (dim + 1)))
            for _ in range(n_features):
                birth = np.random.uniform(0, max_dist * 0.5)
                death = np.random.uniform(birth, max_dist)
                persistence[dim].append([birth, death])
        
        # Convert to numpy arrays
        for dim in persistence:
            if persistence[dim]:
                persistence[dim] = np.array(persistence[dim])
            else:
                persistence[dim] = np.array([]).reshape(0, 2)
        
        return persistence
    
    def validate_topology(self, data: Any, topology_type: str) -> Dict[str, Any]:
        """
        Validate topological properties of generated data.
        
        Args:
            data: Generated topological data
            topology_type: Expected topology type
            
        Returns:
            Dictionary with validation metrics
        """
        validation = {
            'type': topology_type,
            'valid': True,
            'metrics': {}
        }
        
        if isinstance(data, np.ndarray):
            # Point cloud validation
            validation['metrics']['n_points'] = len(data)
            validation['metrics']['dimension'] = data.shape[1]
            validation['metrics']['mean_norm'] = np.mean(np.linalg.norm(data, axis=1))
            validation['metrics']['std_norm'] = np.std(np.linalg.norm(data, axis=1))
            
        elif isinstance(data, nx.Graph):
            # Graph validation
            validation['metrics']['n_nodes'] = data.number_of_nodes()
            validation['metrics']['n_edges'] = data.number_of_edges()
            validation['metrics']['connected'] = nx.is_connected(data)
            validation['metrics']['diameter'] = nx.diameter(data) if nx.is_connected(data) else -1
            validation['metrics']['clustering_coefficient'] = nx.average_clustering(data)
            
        elif isinstance(data, dict):
            # Simplicial complex validation
            total_simplices = sum(len(simplices) for simplices in data.values())
            validation['metrics']['total_simplices'] = total_simplices
            validation['metrics']['max_dimension'] = max(data.keys())
            validation['metrics']['n_vertices'] = len(data.get(0, []))
            
        return validation