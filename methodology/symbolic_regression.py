"""
Symbolic Regression Module for Formula Discovery
=================================================
Discovers mathematical formulas from topological features using genetic programming.
Includes validation, complexity analysis, and interpretability measures.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from collections import defaultdict
import operator
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

# Define mathematical operations
class Operations:
    """Available mathematical operations for symbolic regression."""
    
    UNARY = {
        'neg': (operator.neg, '-'),
        'abs': (abs, 'abs'),
        'sqrt': (lambda x: np.sqrt(np.abs(x)), 'sqrt'),
        'square': (lambda x: x**2, 'sq'),
        'cube': (lambda x: x**3, 'cube'),
        'exp': (lambda x: np.exp(np.clip(x, -10, 10)), 'exp'),
        'log': (lambda x: np.log(np.abs(x) + 1e-10), 'log'),
        'sin': (np.sin, 'sin'),
        'cos': (np.cos, 'cos'),
        'tanh': (np.tanh, 'tanh')
    }
    
    BINARY = {
        'add': (operator.add, '+'),
        'sub': (operator.sub, '-'),
        'mul': (operator.mul, '*'),
        'div': (lambda x, y: x / (y + 1e-10), '/'),
        'pow': (lambda x, y: np.abs(x) ** np.clip(y, -10, 10), '^'),
        'max': (np.maximum, 'max'),
        'min': (np.minimum, 'min')
    }
    
    CONSTANTS = {
        'pi': np.pi,
        'e': np.e,
        'phi': 1.618033988749895,  # Golden ratio
        'sqrt2': np.sqrt(2),
        'one': 1.0,
        'zero': 0.0
    }


@dataclass
class Node:
    """Tree node for representing mathematical expressions."""
    op_type: str  # 'unary', 'binary', 'variable', 'constant'
    op_name: str
    children: List['Node'] = None
    value: Optional[Union[float, int]] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the expression tree."""
        if self.op_type == 'variable':
            return variables.get(self.op_name, np.zeros(len(next(iter(variables.values())))))
        elif self.op_type == 'constant':
            if self.op_name in Operations.CONSTANTS:
                val = Operations.CONSTANTS[self.op_name]
            else:
                val = self.value if self.value is not None else 0.0
            # Return array of constant values
            return np.full(len(next(iter(variables.values()))), val)
        elif self.op_type == 'unary':
            child_val = self.children[0].evaluate(variables)
            op_func = Operations.UNARY[self.op_name][0]
            return op_func(child_val)
        elif self.op_type == 'binary':
            left_val = self.children[0].evaluate(variables)
            right_val = self.children[1].evaluate(variables)
            op_func = Operations.BINARY[self.op_name][0]
            return op_func(left_val, right_val)
        else:
            raise ValueError(f"Unknown operation type: {self.op_type}")
    
    def to_string(self) -> str:
        """Convert expression tree to string formula."""
        if self.op_type == 'variable':
            return self.op_name
        elif self.op_type == 'constant':
            if self.op_name in Operations.CONSTANTS:
                return self.op_name
            else:
                return f"{self.value:.3f}" if self.value is not None else "0"
        elif self.op_type == 'unary':
            op_symbol = Operations.UNARY[self.op_name][1]
            child_str = self.children[0].to_string()
            return f"{op_symbol}({child_str})"
        elif self.op_type == 'binary':
            op_symbol = Operations.BINARY[self.op_name][1]
            left_str = self.children[0].to_string()
            right_str = self.children[1].to_string()
            if op_symbol in ['+', '-', '*', '/']:
                return f"({left_str} {op_symbol} {right_str})"
            else:
                return f"{op_symbol}({left_str}, {right_str})"
    
    def complexity(self) -> int:
        """Compute complexity of the expression tree."""
        if self.op_type in ['variable', 'constant']:
            return 1
        else:
            return 1 + sum(child.complexity() for child in self.children)
    
    def copy(self) -> 'Node':
        """Deep copy of the node."""
        return deepcopy(self)


class SymbolicRegressor:
    """Genetic programming-based symbolic regression."""
    
    def __init__(self, 
                 parsimony_coefficient: float = 0.01,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9):
        """
        Initialize symbolic regressor.
        
        Args:
            parsimony_coefficient: Penalty for complexity
            tournament_size: Size of tournament selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.parsimony_coefficient = parsimony_coefficient
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.best_programs = []
        self.evolution_history = []
        self.variable_names = []
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_complexity: int = 10,
            population_size: int = 100,
            generations: int = 50,
            variable_names: Optional[List[str]] = None,
            verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Discover symbolic formulas using genetic programming.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            max_complexity: Maximum allowed complexity
            population_size: Number of programs in population
            generations: Number of evolution generations
            variable_names: Names for input variables
            verbose: Whether to print progress
            
        Returns:
            List of best formulas with scores
        """
        # Prepare data
        X = np.atleast_2d(X)
        y = np.array(y).flatten()
        
        n_samples, n_features = X.shape
        
        # Set variable names
        if variable_names is None:
            self.variable_names = [f"x{i}" for i in range(n_features)]
        else:
            self.variable_names = variable_names
        
        # Create variable dictionary for evaluation
        variables = {name: X[:, i] for i, name in enumerate(self.variable_names)}
        
        # Initialize population
        population = self._initialize_population(population_size, max_complexity)
        
        # Evolution loop
        best_formula = None
        best_score = -np.inf
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for program in population:
                try:
                    predictions = program.evaluate(variables)
                    mse = np.mean((predictions - y) ** 2)
                    complexity_penalty = self.parsimony_coefficient * program.complexity()
                    fitness = -mse - complexity_penalty  # Higher is better
                    fitness_scores.append(fitness)
                except:
                    fitness_scores.append(-np.inf)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_score:
                best_score = fitness_scores[best_idx]
                best_formula = population[best_idx].copy()
            
            # Log progress
            if verbose and generation % 10 == 0:
                avg_fitness = np.mean(fitness_scores[fitness_scores > -np.inf])
                print(f"Generation {generation}: Best fitness = {best_score:.4f}, "
                      f"Avg fitness = {avg_fitness:.4f}")
            
            # Store history
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': float(best_score),
                'avg_fitness': float(np.mean(fitness_scores[fitness_scores > -np.inf]))
            })
            
            # Create next generation
            new_population = []
            
            # Elitism: Keep best programs
            elite_size = int(population_size * 0.1)
            elite_idx = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_idx:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < population_size:
                if random.random() < self.crossover_rate and len(new_population) < population_size - 1:
                    # Crossover
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # Apply mutations
                    if random.random() < self.mutation_rate:
                        child1 = self._mutate(child1, max_complexity)
                    if random.random() < self.mutation_rate:
                        child2 = self._mutate(child2, max_complexity)
                    
                    new_population.append(child1)
                    if len(new_population) < population_size:
                        new_population.append(child2)
                else:
                    # Mutation only
                    parent = self._tournament_selection(population, fitness_scores)
                    child = self._mutate(parent.copy(), max_complexity)
                    new_population.append(child)
            
            population = new_population
        
        # Final evaluation and ranking
        final_formulas = []
        for program in population:
            try:
                predictions = program.evaluate(variables)
                
                # Calculate metrics
                mse = np.mean((predictions - y) ** 2)
                rmse = np.sqrt(mse)
                
                # R² score
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10))
                
                # Adjusted R²
                n = len(y)
                p = program.complexity()
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
                
                final_formulas.append({
                    'expression': program.to_string(),
                    'tree': program,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'adjusted_r2': float(adj_r2),
                    'complexity': program.complexity(),
                    'fitness': float(-mse - self.parsimony_coefficient * program.complexity())
                })
            except:
                continue
        
        # Sort by fitness
        final_formulas.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Store best programs
        self.best_programs = final_formulas[:10]
        
        return self.best_programs
    
    def predict(self, X: np.ndarray, formula_idx: int = 0) -> np.ndarray:
        """
        Make predictions using discovered formula.
        
        Args:
            X: Input features
            formula_idx: Index of formula to use (0 = best)
            
        Returns:
            Predictions
        """
        if not self.best_programs:
            raise ValueError("No formulas discovered yet. Run fit() first.")
        
        if formula_idx >= len(self.best_programs):
            raise ValueError(f"Formula index {formula_idx} out of range")
        
        X = np.atleast_2d(X)
        variables = {name: X[:, i] for i, name in enumerate(self.variable_names)}
        
        program = self.best_programs[formula_idx]['tree']
        return program.evaluate(variables)
    
    def validate_formulas(self,
                         formulas: List[Dict[str, Any]],
                         X: np.ndarray,
                         y: np.ndarray,
                         cv_folds: int = 5) -> Dict[str, Any]:
        """
        Validate discovered formulas using cross-validation.
        
        Args:
            formulas: List of formula dictionaries
            X: Input features
            y: Target values
            cv_folds: Number of cross-validation folds
            
        Returns:
            Validation results
        """
        from sklearn.model_selection import KFold
        
        validation_results = []
        
        for formula in formulas[:5]:  # Validate top 5 formulas
            program = formula['tree']
            
            # Cross-validation
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Evaluate on test set
                variables_test = {name: X_test[:, i] 
                                for i, name in enumerate(self.variable_names)}
                
                try:
                    predictions = program.evaluate(variables_test)
                    mse = np.mean((predictions - y_test) ** 2)
                    cv_scores.append(mse)
                except:
                    cv_scores.append(np.inf)
            
            validation_results.append({
                'expression': formula['expression'],
                'cv_mse_mean': float(np.mean(cv_scores)),
                'cv_mse_std': float(np.std(cv_scores)),
                'training_r2': formula['r2'],
                'complexity': formula['complexity'],
                'generalization_score': formula['r2'] / (1 + formula['complexity'] * 0.01)
            })
        
        return {
            'validated_formulas': validation_results,
            'best_generalizing': max(validation_results, 
                                    key=lambda x: x['generalization_score'])
        }
    
    def analyze_complexity(self, formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze complexity of discovered formulas.
        
        Args:
            formulas: List of formula dictionaries
            
        Returns:
            Complexity analysis
        """
        if not formulas:
            return {'error': 'No formulas to analyze'}
        
        complexities = [f['complexity'] for f in formulas]
        r2_scores = [f['r2'] for f in formulas]
        
        # Pareto frontier analysis
        pareto_frontier = []
        for i, (comp, r2) in enumerate(zip(complexities, r2_scores)):
            # Check if dominated
            is_dominated = False
            for j, (comp2, r2_2) in enumerate(zip(complexities, r2_scores)):
                if i != j and comp2 <= comp and r2_2 >= r2:
                    if comp2 < comp or r2_2 > r2:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_frontier.append(i)
        
        # Analyze trade-off
        if len(pareto_frontier) > 1:
            frontier_formulas = [formulas[i] for i in pareto_frontier]
            frontier_formulas.sort(key=lambda x: x['complexity'])
            
            # Compute elbow point (best trade-off)
            if len(frontier_formulas) >= 3:
                # Use angle-based method
                angles = []
                for i in range(1, len(frontier_formulas) - 1):
                    prev = frontier_formulas[i-1]
                    curr = frontier_formulas[i]
                    next = frontier_formulas[i+1]
                    
                    # Compute angle
                    v1 = np.array([prev['complexity'] - curr['complexity'],
                                 prev['r2'] - curr['r2']])
                    v2 = np.array([next['complexity'] - curr['complexity'],
                                 next['r2'] - curr['r2']])
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)
                
                best_tradeoff_idx = np.argmin(angles) + 1
                best_tradeoff = frontier_formulas[best_tradeoff_idx]
            else:
                best_tradeoff = frontier_formulas[0]
        else:
            best_tradeoff = formulas[0] if formulas else None
        
        return {
            'complexity_range': [min(complexities), max(complexities)],
            'r2_range': [min(r2_scores), max(r2_scores)],
            'pareto_frontier_size': len(pareto_frontier),
            'pareto_formulas': [formulas[i] for i in pareto_frontier],
            'best_tradeoff': best_tradeoff,
            'complexity_statistics': {
                'mean': float(np.mean(complexities)),
                'std': float(np.std(complexities)),
                'median': float(np.median(complexities))
            }
        }
    
    def extract_important_variables(self, formula: Dict[str, Any]) -> List[str]:
        """
        Extract important variables from a formula.
        
        Args:
            formula: Formula dictionary
            
        Returns:
            List of important variable names
        """
        program = formula['tree']
        variables = set()
        
        def extract_vars(node):
            if node.op_type == 'variable':
                variables.add(node.op_name)
            for child in node.children:
                extract_vars(child)
        
        extract_vars(program)
        return sorted(list(variables))
    
    def simplify_formula(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify a formula by removing redundant operations.
        
        Args:
            formula: Formula dictionary
            
        Returns:
            Simplified formula
        """
        program = formula['tree'].copy()
        
        def simplify_node(node):
            # Recursively simplify children
            for i, child in enumerate(node.children):
                node.children[i] = simplify_node(child)
            
            # Apply simplification rules
            if node.op_type == 'binary':
                if node.op_name == 'add':
                    # x + 0 = x
                    if (node.children[1].op_type == 'constant' and 
                        node.children[1].value == 0):
                        return node.children[0]
                    if (node.children[0].op_type == 'constant' and 
                        node.children[0].value == 0):
                        return node.children[1]
                
                elif node.op_name == 'mul':
                    # x * 1 = x
                    if (node.children[1].op_type == 'constant' and 
                        node.children[1].value == 1):
                        return node.children[0]
                    if (node.children[0].op_type == 'constant' and 
                        node.children[0].value == 1):
                        return node.children[1]
                    # x * 0 = 0
                    if ((node.children[0].op_type == 'constant' and 
                         node.children[0].value == 0) or
                        (node.children[1].op_type == 'constant' and 
                         node.children[1].value == 0)):
                        return Node('constant', 'zero')
            
            return node
        
        simplified = simplify_node(program)
        
        return {
            'expression': simplified.to_string(),
            'tree': simplified,
            'complexity': simplified.complexity(),
            'original_complexity': formula['complexity']
        }
    
    def _initialize_population(self, size: int, max_complexity: int) -> List[Node]:
        """Initialize random population of programs."""
        population = []
        
        for _ in range(size):
            # Random complexity between 1 and max_complexity
            target_complexity = random.randint(1, max_complexity)
            program = self._generate_random_program(target_complexity)
            population.append(program)
        
        return population
    
    def _generate_random_program(self, target_complexity: int) -> Node:
        """Generate a random program with approximate target complexity."""
        if target_complexity <= 1:
            # Terminal node
            if random.random() < 0.5:
                # Variable
                var_name = random.choice(self.variable_names) if self.variable_names else 'x0'
                return Node('variable', var_name)
            else:
                # Constant
                if random.random() < 0.3:
                    const_name = random.choice(list(Operations.CONSTANTS.keys()))
                    return Node('constant', const_name)
                else:
                    value = random.uniform(-5, 5)
                    return Node('constant', 'value', value=value)
        
        # Non-terminal node
        if random.random() < 0.3:
            # Unary operation
            op_name = random.choice(list(Operations.UNARY.keys()))
            child = self._generate_random_program(target_complexity - 1)
            return Node('unary', op_name, children=[child])
        else:
            # Binary operation
            op_name = random.choice(list(Operations.BINARY.keys()))
            # Split complexity between children
            left_complexity = random.randint(1, max(1, target_complexity - 2))
            right_complexity = target_complexity - 1 - left_complexity
            
            left_child = self._generate_random_program(left_complexity)
            right_child = self._generate_random_program(right_complexity)
            
            return Node('binary', op_name, children=[left_child, right_child])
    
    def _tournament_selection(self, population: List[Node], fitness: np.ndarray) -> Node:
        """Select individual using tournament selection."""
        tournament_idx = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = fitness[tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        """Perform crossover between two programs."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Get all nodes
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)
        
        if len(nodes1) > 1 and len(nodes2) > 1:
            # Select random crossover points
            point1 = random.choice(nodes1[1:])  # Skip root
            point2 = random.choice(nodes2[1:])
            
            # Swap subtrees
            temp_children = point1.children
            temp_op = point1.op_name
            temp_type = point1.op_type
            temp_value = point1.value
            
            point1.children = point2.children
            point1.op_name = point2.op_name
            point1.op_type = point2.op_type
            point1.value = point2.value
            
            point2.children = temp_children
            point2.op_name = temp_op
            point2.op_type = temp_type
            point2.value = temp_value
        
        return child1, child2
    
    def _mutate(self, program: Node, max_complexity: int) -> Node:
        """Apply mutation to a program."""
        mutated = program.copy()
        nodes = self._get_all_nodes(mutated)
        
        if not nodes:
            return mutated
        
        # Select random node to mutate
        node = random.choice(nodes)
        
        # Mutation type
        mutation_type = random.choice(['replace', 'insert', 'delete'])
        
        if mutation_type == 'replace':
            # Replace with random subtree
            new_subtree = self._generate_random_program(min(3, max_complexity))
            node.op_type = new_subtree.op_type
            node.op_name = new_subtree.op_name
            node.children = new_subtree.children
            node.value = new_subtree.value
        
        elif mutation_type == 'insert' and node.op_type in ['variable', 'constant']:
            # Insert operation above terminal
            op_type = random.choice(['unary', 'binary'])
            if op_type == 'unary':
                op_name = random.choice(list(Operations.UNARY.keys()))
                old_node = Node(node.op_type, node.op_name, value=node.value)
                node.op_type = 'unary'
                node.op_name = op_name
                node.children = [old_node]
                node.value = None
            else:
                op_name = random.choice(list(Operations.BINARY.keys()))
                old_node = Node(node.op_type, node.op_name, value=node.value)
                new_node = self._generate_random_program(1)
                node.op_type = 'binary'
                node.op_name = op_name
                node.children = [old_node, new_node] if random.random() < 0.5 else [new_node, old_node]
                node.value = None
        
        elif mutation_type == 'delete' and node.op_type == 'unary':
            # Remove unary operation
            if node.children:
                child = node.children[0]
                node.op_type = child.op_type
                node.op_name = child.op_name
                node.children = child.children
                node.value = child.value
        
        # Limit complexity
        if mutated.complexity() > max_complexity:
            return program  # Return original if too complex
        
        return mutated
    
    def _get_all_nodes(self, program: Node) -> List[Node]:
        """Get all nodes in the program tree."""
        nodes = [program]
        for child in program.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes