import os
import json
import numpy as np

# agent class to string mapping that is used to select the agent class in parameters.json
from agent import BallisticAgent, LévyAgent, BrownianAgent, ReservoirAgent, ExponentialAgent
agent_classes = {
    "ballistic": BallisticAgent,
    "levy": LévyAgent,
    "exponential": ExponentialAgent,
    "brownian": BrownianAgent,
    "reservoir": ReservoirAgent
}

class Params:
    """
    Class that stores parameters of the simulation.
    """
    def __init__(self, **kwargs):
        # map agent type to class
        agent_type_str = kwargs.get('type')
        self.agent = self._get_agent_class(agent_type_str)

        for key, value in kwargs.items():
            if key != 'type': 
                setattr(self, key, value)

        # one more for the initial positions
        self.simulation_steps = len(np.arange(0, self.total_time, self.delta_t)) + 1

        if self.mu <= 1:
            raise ValueError("Parameter 'mu' must be greater than 1.")
        if self.mu > 3:
            raise ValueError("Parameter 'mu' must be less than or equal to 3.")
        if self.alpha <= 0:
            raise ValueError("Parameter 'alpha' must be greater than 0.")
        
        if self.mu < 1.5:
            print("Warning: Value for mu is too low, simulation may crash.")
        
    def _get_agent_class(self, agent_type_str):
        return agent_classes.get(agent_type_str)

    @classmethod
    def from_json(cls, file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        flat_data = {**data['agent'], **data['environment'], **data['simulation']}
        
        return cls(**flat_data)
