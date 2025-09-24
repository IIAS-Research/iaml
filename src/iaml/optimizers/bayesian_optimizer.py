"""
Bayesian Optimization-based Optimizer for IAML.
Optimizer receives a pool of Candidates, optimizes parameters using a surrogate model,
and returns a new pool of candidates.
"""
from copy import deepcopy
import random
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from ..candidate import Candidate
from .optimizer import Optimizer as BaseOptimizer
from ..step import Step
from ..logger import Logger

class BayesianOptimizer(BaseOptimizer):
    def __init__(self, duration: int = None, max_iterations=50):
        """
        Initialize the Bayesian Optimization-based Optimizer.
        :param max_iterations: Number of optimization iterations.
        """
        self.max_iterations = max_iterations
        self.ignored_configs: list[str] = {'random_state'}
        self.current_iteration = 0
        self.max_candidates = 40
        self.skopt_optimizers = {}
    
    def _generate_structure_id(self, candidate):
        """Generate a unique identifier for the candidate structure."""
        return hash(tuple((step[0], tuple(sorted(step[1].configuration.keys()))) for step in candidate.pipeline.steps))
    
    def _initialize_search_space(self, candidates):
        """Define the search space for each unique candidate structure."""
        for candidate in candidates:
            structure_id = self._generate_structure_id(candidate)
            if structure_id in self.skopt_optimizers:
                continue  # Avoid reinitializing existing structures
            
            dimensions = []
            param_keys = []
            
            for step_idx, step in enumerate(candidate.pipeline.steps):
                for key, config in sorted(step[1].configuration.items()):
                    if key in self.ignored_configs:
                        continue
                    param_key = f"{step_idx}_{key}"
                    param_keys.append(param_key)
                    
                    value = config['value']
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    
                    if isinstance(value, (int, float)):
                        if 'range' in config:
                            low, high = config['range']
                            if low > high:
                                low, high = high, low
                            if isinstance(value, float):
                                dimensions.append(Real(low, high))
                            else:
                                if low <= 0:
                                    low = 1
                                dimensions.append(Integer(low, high))
                        else:
                            dimensions.append(Real(-1e6, 1e6) if isinstance(value, float) else Integer(-1e6, 1e6))
                    elif 'categorical' in config.keys():
                        dimensions.append(Categorical(config['categorical']))
                    elif isinstance(value, bool):
                        dimensions.append(Categorical([True, False]))
            
            Logger().info(f"Initializing Bayesian Optimizer for structure {structure_id} with {len(dimensions)} dimensions")
            self.skopt_optimizers[structure_id] = {'optimizer': Optimizer(dimensions), 'param_keys': param_keys, 'dimensions': dimensions}
    
    def _suggest_new_candidates(self, candidates):
        """Suggest a new set of candidates using Bayesian Optimization."""
        new_candidates = []
        for candidate in candidates:
            structure_id = self._generate_structure_id(candidate)
            optimizer_data = self.skopt_optimizers.get(structure_id)
            if not optimizer_data:
                Logger().warning(f"Missing optimizer for structure {structure_id}")
                continue
            
            new_params = optimizer_data['optimizer'].ask(n_points=1)[0]
            if len(new_params) != len(optimizer_data['param_keys']):
                Logger().error(f"Parameter mismatch: expected {len(optimizer_data['param_keys'])}, got {len(new_params)}")
                continue
            
            new_candidate = deepcopy(candidate)
            idx = 0
            
            for step_idx, step in enumerate(new_candidate.pipeline.steps):
                for key, config in sorted(step[1].configuration.items()):
                    if key in self.ignored_configs:
                        continue
                    param_key = f"{step_idx}_{key}"
                    if param_key in optimizer_data['param_keys']:
                        if isinstance(config['value'], bool):
                            step[1].configure(key, bool(new_params[idx]))
                        elif isinstance(new_params[idx], (np.integer, np.floating)):
                            step[1].configure(key, new_params[idx].item())
                        else:
                            step[1].configure(key, new_params[idx])
                        idx += 1
            new_candidates.append(new_candidate)
        
        return new_candidates
    
    def run(self, candidates):
        """
        Perform Bayesian Optimization iteration.
        :param candidates: List of Candidate objects to optimize.
        :return: New list of candidates optimized using Bayesian Optimization.
        """
        if self.current_iteration == 0:
            self._initialize_search_space(candidates)
        
        candidates.sort(key=lambda c: c.get_main_metric_value(), reverse=True)
        candidates = candidates[0:self.max_candidates//2]
        
        for candidate in candidates:
            structure_id = self._generate_structure_id(candidate)
            optimizer_data = self.skopt_optimizers.get(structure_id)
            if not optimizer_data:
                Logger().warning(f"Skipping candidate {structure_id}, missing optimizer")
                continue
            
            params = []
            for step_idx, step in enumerate(candidate.pipeline.steps):
                for key, config in sorted(step[1].configuration.items()):
                    if key in self.ignored_configs:
                        continue
                    param_key = f"{step_idx}_{key}"
                    if param_key in optimizer_data['param_keys']:
                        try:
                            param_index = optimizer_data['param_keys'].index(param_key)
                            value = config['value']
                            if isinstance(value, (np.integer, np.floating)):
                                value = value.item()
                            
                            bounds = optimizer_data['dimensions'][param_index].bounds
                            if isinstance(bounds, tuple) and isinstance(value, (int, float)):
                                value = max(min(value, bounds[1]), bounds[0])
                            params.append(value)
                        except IndexError as e:
                            Logger().error(f"IndexError: {str(e)} - param_key: {param_key}, param_keys: {optimizer_data['param_keys']}")
                            continue
            
            if len(params) != len(optimizer_data['param_keys']):
                Logger().error(f"Mismatch in parameters: expected {len(optimizer_data['param_keys'])}, got {len(params)}")
                continue
            
            main_metric = candidate.get_main_metric_value()
            if isinstance(main_metric, (np.integer, np.floating)):
                main_metric = main_metric.item()
            if not isinstance(main_metric, (int, float)):
                Logger().error(f"Invalid main_metric: expected scalar, got {type(main_metric)} - {main_metric}")
                continue
            
            try:
                optimizer_data['optimizer'].tell([params], [main_metric])
            except ValueError as e:
                Logger().error(f"Error in skopt.tell(): {str(e)}. Params: {params}, Bounds: {optimizer_data['optimizer'].space.bounds}")
                continue
        
        new_candidates = self._suggest_new_candidates(candidates)
        self.current_iteration += 1
        return new_candidates + candidates
    
    @property
    def finished(self) -> bool:
        """
        Check if optimization is finished.
        :return: Boolean indicating if optimization is complete.
        """
        return self.current_iteration >= self.max_iterations