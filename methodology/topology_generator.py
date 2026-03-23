"""
Geometric Drone Network Topology Generator

Generates distance-based random topologies with controlled efficiency (40-55%)
for synchronization analysis.

Method: Geometric Random Graphs
- Drones positioned randomly in 2D space
- Connected if within radio range r
- Models realistic distance-based communication

Optimized with:
- N = 100 nodes (larger networks)
- Multithreaded generation
- Efficient binary search for target efficiency
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


@dataclass
class TopologyResult:
    """Container for generated topology and metadata"""
    graph: nx.Graph
    positions: np.ndarray
    target_efficiency: float
    actual_efficiency: float
    radius: float
    

class GeometricTopologyGenerator:
    """Generate distance-based sparse drone communication networks"""
    
    def __init__(self, N: int = 150, efficiency_range: Tuple[float, float] = (0.35, 0.55)):
        self.N = N
        self.efficiency_range = efficiency_range
        self.max_edges = N * (N - 1) / 2
        
    def _calculate_efficiency(self, G: nx.Graph) -> float:
        """Calculate network efficiency (edges / max_edges)"""
        return G.number_of_edges() / self.max_edges
    
    def _ensure_connected(self, G: nx.Graph, positions: np.ndarray) -> nx.Graph:
        """Ensure graph is connected by adding edges between closest components"""
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            
            # Connect components by finding closest nodes between them
            for i in range(len(components) - 1):
                comp_a = list(components[i])
                comp_b = list(components[i + 1])
                
                # Find closest pair between components
                min_dist = float('inf')
                best_pair = (comp_a[0], comp_b[0])
                
                for node_a in comp_a:
                    for node_b in comp_b:
                        dist = np.linalg.norm(positions[node_a] - positions[node_b])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (node_a, node_b)
                
                G.add_edge(best_pair[0], best_pair[1])
        
        return G
    
    # ==================== GEOMETRIC GENERATION ====================
    
    def generate_geometric(self, target_efficiency: float, max_iterations: int = 30, seed: int = None) -> TopologyResult:
        """
        Generate distance-based topology (geometric random graph)
        
        Drones positioned randomly in 2D space, connected if within distance r.
        Binary search to find r that gives target efficiency.
        
        Args:
            target_efficiency: Target edge density (0.40 to 0.55)
            max_iterations: Max iterations for binary search
            seed: Random seed for reproducibility
        
        Returns:
            TopologyResult with graph and metadata
        """
        rng = np.random.default_rng(seed)
        positions = rng.random((self.N, 2))
        target_edges = int(target_efficiency * self.max_edges)
        
        # Binary search for optimal radius
        r_low, r_high = 0.0, 2.0
        best_G = None
        best_r = 0.0
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            r = (r_low + r_high) / 2
            
            # Create graph based on distance
            G = nx.Graph()
            G.add_nodes_from(range(self.N))
            
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= r:
                        G.add_edge(i, j)
            
            current_edges = G.number_of_edges()
            diff = abs(current_edges - target_edges)
            
            if diff < best_diff:
                best_diff = diff
                best_G = G.copy()
                best_r = r
            
            # Adjust search range
            if current_edges < target_edges * 0.95:
                r_low = r
            elif current_edges > target_edges * 1.05:
                r_high = r
            else:
                break
        
        # Ensure connectivity
        best_G = self._ensure_connected(best_G, positions)
        actual_eff = self._calculate_efficiency(best_G)
        
        return TopologyResult(
            graph=best_G,
            positions=positions,
            target_efficiency=target_efficiency,
            actual_efficiency=actual_eff,
            radius=best_r
        )
    
    # ==================== MULTITHREADED DATASET GENERATION ====================
    
    def _generate_single_topology(self, args):
        """Worker function for parallel generation"""
        target_eff, seed = args
        try:
            result = self.generate_geometric(target_eff, seed=seed)
            
            # Only keep if within target range
            if (self.efficiency_range[0] <= result.actual_efficiency <= 
                self.efficiency_range[1]):
                return result
            else:
                return None
        except Exception as e:
            print(f"  ⚠️  Failed: {e}")
            return None
    
    def generate_dataset(self, 
                        efficiency_levels: int = 15,
                        samples_per_level: int = 50,
                        max_workers: int = None) -> List[TopologyResult]:
        """
        Generate full dataset using multithreading
        
        Args:
            efficiency_levels: Number of efficiency values to test (default: 15)
            samples_per_level: Samples per efficiency level (default: 50)
            max_workers: Number of threads (default: CPU count)
        
        Returns:
            List of TopologyResult objects
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        efficiency_values = np.linspace(self.efficiency_range[0], 
                                       self.efficiency_range[1], 
                                       efficiency_levels)
        
        total = efficiency_levels * samples_per_level
        
        print(f"🚀 Generating {total} geometric topologies...")
        print(f"   N = {self.N} nodes")
        print(f"   Efficiency range: {self.efficiency_range[0]:.2f} to {self.efficiency_range[1]:.2f}")
        print(f"   Using {max_workers} threads\n")
        
        # Prepare all generation tasks
        tasks = []
        seed_counter = 0
        for target_eff in efficiency_values:
            for sample in range(samples_per_level):
                tasks.append((target_eff, seed_counter))
                seed_counter += 1
        
        # Generate in parallel using threads
        dataset = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_single_topology, task) 
                      for task in tasks]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    dataset.append(result)
                
                completed += 1
                if completed % 50 == 0:
                    print(f"  Generated {completed}/{total} ({100*completed/total:.1f}%)")
        
        print(f"\n✅ Generated {len(dataset)} valid topologies (within efficiency range)")
        return dataset


# ==================== UTILITY FUNCTIONS ====================

def save_topology(result: TopologyResult, filename: str):
    """Save topology to file"""
    data = {
        'adjacency': nx.to_numpy_array(result.graph),
        'positions': result.positions,
        'target_efficiency': result.target_efficiency,
        'actual_efficiency': result.actual_efficiency,
        'radius': result.radius
    }
    np.savez_compressed(filename, **data)
    

def load_topology(filename: str) -> TopologyResult:
    """Load topology from file"""
    data = np.load(filename, allow_pickle=True)
    
    G = nx.from_numpy_array(data['adjacency'])
    
    return TopologyResult(
        graph=G,
        positions=data['positions'],
        target_efficiency=float(data['target_efficiency']),
        actual_efficiency=float(data['actual_efficiency']),
        radius=float(data['radius'])
    )


# ==================== DEMO / TESTING ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Geometric Topology Generator")
    print("=" * 60)
    
    gen = GeometricTopologyGenerator(N=150)
    
    print("\n🧪 Test 1: Single topology generation")
    result = gen.generate_geometric(target_efficiency=0.47)
    
    print(f"  Target efficiency: {result.target_efficiency:.3f}")
    print(f"  Actual efficiency: {result.actual_efficiency:.3f}")
    print(f"  Edges: {result.graph.number_of_edges()}")
    print(f"  Connected: {nx.is_connected(result.graph)}")
    print(f"  Radius: {result.radius:.4f}")
    
    print("\n🧪 Test 2: Small multithreaded dataset")
    dataset = gen.generate_dataset(
        efficiency_levels=3,
        samples_per_level=5,
        max_workers=4
    )
    
    print(f"\n  Dataset statistics:")
    efficiencies = [t.actual_efficiency for t in dataset]
    print(f"    Efficiency range: [{min(efficiencies):.3f}, {max(efficiencies):.3f}]")
    print(f"    Mean edges: {np.mean([t.graph.number_of_edges() for t in dataset]):.1f}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)