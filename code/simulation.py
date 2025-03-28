import numpy as np
from environment import Environment
from agent import *
from evolution import EvolutionaryAlgorithm
import os
import json
from data_io import update_epoch_data, initialize_epoch_data, save_simulation_context, save_epoch_data
import config
import time
from tqdm import tqdm

class Simulation:
    """
    Collection of methods that are used to run a simulation.
    """

    def __init__(self, params, agent):
        """
        Args:
            params (Params): class that stores parameters of the simulation
            agent (Class): agent type that is used in the simulation
        """
        self.params = params
        self.agent = agent
        self.total_time = params.total_time
        self.delta_t = params.delta_t
        self.num_epochs = params.num_epochs
        self.iterations_per_epoch = params.iterations_per_epoch
        self.population_size = params.population_size
        
        self.iteration = 0
        # per iteration store the position, direction, and whether agent consumed food at each time step
        self.trajectory_log = np.zeros((self.params.simulation_steps, self.population_size, config.NUM_MOTION_ATTRIBUTES))

        self.data = initialize_epoch_data(self.params)

    def run(self, folder, population=None):
        """
        Run a simulation.

        Each epoch:
        - evaluates the performance of each agent iterations_per_epoch times where
            Each iteration:
            - simulates the agent behaviour for simulation_steps time steps
        - evolves the population

        Args:
            folder (str): store the simulation results in
            population (list): list of agents
        """
        environment = Environment(self.params)
        population = self.set_up_population(population)

        print('Starting simulation...')
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            population = self.run_epoch(population, environment)
            # TODO implement when fixing evolution
            # if epoch % config.INTERVALL_SAVE == 0:
            #    save_epoch_data(folder, self.data, population, epoch)
        print('Saving simulation...')
        save_simulation_context(folder, environment, self.params)
        save_epoch_data(folder, self.data, population, self.params.num_epochs)

    def set_up_population(self, population):
        """
        Either use an existing population or create a population from scratch by instantiating
        agents with the agent class.
        """
        if population:
            return population  
        return [self.agent(self.params) for _ in range(self.population_size)]
    
    def record_iteration_data(func):
        """
        Decorator to be called after each iteration.
        - keeps track of the meals each agent consumed in the iteration
        - stores the movement data of each agent
        """
        def wrapper(self, population, environment):
            result = func(self, population, environment)
            update_epoch_data(self.data, self.iteration, self.trajectory_log)
            return result
        return wrapper

    def record_move(func):
        """
        Decorator to store movement data of each agent at the time step.        
        """
        def wrapper(self, population, environment, *args, **kwargs):
            result = func(self, population, environment, *args, **kwargs)
            self.collect_data(population, kwargs['step'])
            return result
        return wrapper

    def run_epoch(self, population, environment):
        """
        Trains the population for a single epoch.
        - simulate agent foraging for all iterations
        - evaluate the performance of each agent
        - evolve the population
        Returns the next generation.
        """
        self.iteration = 0
        for _ in range(self.iterations_per_epoch):
            self.run_iteration(population, environment)
            self.iteration += 1
        # sum up consumed food particles over all iterations and time steps
        self.fitnesses = np.asarray(np.sum(self.data['ate'], axis=(0,1)))
        descendants = self.evolve(population) if self.params.evolve else population
        return descendants
    
    @record_iteration_data
    def run_iteration(self, population, environment):
        """
        Simulate the agent behaviour for a certain time in a single iteration.
        """
        self.recycle_agents(population, environment, step=0)
        for step in range(self.params.simulation_steps - 1):
            self.simulate_step(population, environment, step=step + 1)

    @record_move
    def recycle_agents(self, population, environment, *args, **kwargs):
        """
        Reset agents so they can be reused in the next iteration.
        """
        for agent in population:
            agent.reset()

    @record_move
    def simulate_step(self, population, environment, *args, **kwargs):
        """
        Simulates agent action for one time step.
        Agents:
            - perceive the environment
            - choose an action
            - perform the action.
        This logic is followed by every agent.
        """
        for agent in population:
            perception = agent.perceive(environment)
            action = agent.choose_action(perception)
            agent.perform_action(environment, action)
    
    def evolve(self, population):
        """
        Exchange old population with a new population.
        """
        evolutionary_algorithm = EvolutionaryAlgorithm(self.params)
        # sort population after performance in descending order
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        new_population = evolutionary_algorithm.evolve(sorted_population) 
        return new_population

    def collect_data(self, population, step):
        """
        Logs the position, direction and whether the agent consumed food for each agent at the given step.
        """
        for i, agent in enumerate(population):
            self.trajectory_log[step, i] = [agent.position[0], agent.position[1], agent.direction, False]
            self.trajectory_log[step - 1, i, 3] = agent.ate

    
# not currently used
def record_time(func):
    """
    Decorator that adds to a function to log the time it takes.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('Time:', time.time() - start)
        return result
    return wrapper

class Params:
    """
    Class that stores parameters of the simulation.
    """
    def __init__(self, **kwargs):
        # map agent type to class
        agent_type_str = kwargs.get('type')
        self.agent = self._get_agent_class(agent_type_str)

        # only RnnAgents can evolve
        self.evolve = self.agent == RnnAgent

        for key, value in kwargs.items():
            if key != 'type': 
                setattr(self, key, value)
        
        # one more for the initial positions
        self.simulation_steps = len(np.arange(0, self.total_time, self.delta_t)) + 1

    def _get_agent_class(self, agent_type_str):
        agent_classes = {
            "RnnAgent": RnnAgent,
            "BallisticAgent": BallisticAgent,
            "LevyAgent": LévyAgent
        }
        return agent_classes.get(agent_type_str)

    @classmethod
    def from_json(cls, file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        flat_data = {**data['agent'], **data['environment'], 
                     **data['evolution'], **data['simulation']}
        
        return cls(**flat_data)


if __name__ == '__main__':
    print('')