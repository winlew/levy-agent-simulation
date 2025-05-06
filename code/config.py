from pathlib import Path
import multiprocessing as mp
import json
import os
import numpy as np

# x, y, direction, boolean whether agent ate 
NUM_MOTION_ATTRIBUTES = 4

PROJECT_ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT_PATH / 'data'

MAX_FOOD_GENERATION_ATTEMPTS = 1000

MAX_PROCESSES = min(mp.cpu_count(), 5)

from agent import RnnAgent, BallisticAgent, LévyAgent, BrownianAgent, ReservoirAgent

agent_classes = {
    "rnn": RnnAgent,
    "ballistic": BallisticAgent,
    "levy": LévyAgent,
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

        # only RnnAgents can evolve
        if self.agent == agent_classes['rnn']:
            self.evolve = True
        else:
            self.evolve = False
            if self.num_epochs > 1:
                raise UserWarning('Evolution is only supported for RnnAgents. Set num_epochs to 1.')
        
        # one more for the initial positions
        self.simulation_steps = len(np.arange(0, self.total_time, self.delta_t)) + 1

    def _get_agent_class(self, agent_type_str):
        return agent_classes.get(agent_type_str)

    @classmethod
    def from_json(cls, file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        flat_data = {**data['agent'], **data['environment'], 
                     **data['evolution'], **data['simulation'], **data['settings']}
        
        return cls(**flat_data)
