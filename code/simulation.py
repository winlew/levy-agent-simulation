import numpy as np
from environment import Environment
from agent import Agent, Rnn
from evolution import EvolutionaryAlgorithm
import os
import json
from data_io import update_epoch_data, initialize_epoch_data, save_simulation_context, load_population, update_fitness_log, save_epoch_data, write_parameters_to_text
import config
import time
from tqdm import tqdm

class Simulation:

    def __init__(self, params):
        """
        Args:
            params (Params): class that stores parameters of the simulation
                total_time (int): total time of the simulation
                delta_t (int): time step increment
                num_epochs (int): number of epochs
                iterations_per_epoch (int): number of iterations per epoch
                population_size (int): number of agents in the population
        """
        self.params = params
        self.total_time = params.total_time
        self.delta_t = params.delta_t
        self.num_epochs = params.num_epochs
        self.iterations_per_epoch = params.iterations_per_epoch
        self.population_size = params.population_size
        
        self.iteration = 0
        self.data = initialize_epoch_data(self.params)
        # per epoch:
        # - keep track of all food items consumend by each agent in each iteration and
        # - the mean fitness per agent across all iterations
        self.meals_per_iteration = np.zeros((self.iterations_per_epoch, self.population_size))
        self.fitnesses = np.zeros(self.population_size)
        # per iteration:
        # - store the position, direction, and perception radius of each agent at each time step
        self.trajectory_log = np.zeros((self.params.simulation_steps, self.population_size, config.NUM_MOTION_ATTRIBUTES))

        # mean fitness of the population over all iterations for every epoch
        self.population_fitness_log = []


    def run(self, folder, load_folder=None):
        """
        Run a simulation that is configured by the parameter object.

        Each epoch:
        - evaluates the performance of each agent iterations_per_epoch times where
            Each iteration:
            - simulates the agent behaviour for simulation_steps time steps
        - evolves the population

        Args:
            folder (str): store the simulation results in
            load_folder (str): load the population from
        """
        environment = Environment(self.params)
        model = Rnn(self.params)
        population = self.set_up_population(model, load_folder)

        print('Starting simulation...')
        for epoch in tqdm(range(1, self.num_epochs + 1)):
            population = self.run_epoch(population, environment, folder)
            if epoch % config.INTERVALL_SAVE == 0:
                save_epoch_data(folder, self.data, population, epoch)
        save_simulation_context(folder, environment, self.params)
        update_fitness_log(self.data, self.population_fitness_log)
        save_epoch_data(folder, self.data, population, self.params.num_epochs)
        write_parameters_to_text(self.params, folder)

    def set_up_population(self, model, load_folder):
        """
        Load an existing population or create a new one.
        """
        population = []
        if load_folder:
            population = load_population(load_folder)
        else:
            for _ in range(self.population_size):
                agent = Agent(model, self.params)
                population.append(agent)
        return population

    def run_epoch(self, population, environment, folder):
        """
        Trains the population for a single epoch.
        - simulate agent foraging for all iterations
        - evaluate the performance of each agent
        - evolve the population
        """
        self.iteration = 0
        for _ in range(self.iterations_per_epoch):
            self.run_iteration(population, environment)
            self.iteration += 1
        self.fitnesses = np.mean(self.meals_per_iteration, axis=0)
        self.population_fitness_log.append(np.sum(self.fitnesses))
        descendants = self.evolve(population)
        return descendants
    
    def record_iteration_data(func):
        """
        Decorator to be called after each iteration.
        - keeps track of the meals each agent consumed in the iteration
        - stores the movement data of each agent
        """
        def wrapper(self, population, environment):
            result = func(self, population, environment)
            self.store_consumed_meals(population)
            update_epoch_data(self.data, self.iteration, self.trajectory_log, self.meals_per_iteration)
            return result
        return wrapper
    
    @record_iteration_data
    def run_iteration(self, population, environment):
        """
        Simulate the agent behaviour for a certain time.
        """
        self.recycle_agents(population, environment, step=0)
        for step in range(self.params.simulation_steps - 1):
            self.simulate_step(population, environment, step=step+1)
    
    def record_move(func):
        """
        Decorator to store movement data of each agent at the time step.        
        """
        def wrapper(self, population, environment, *args, **kwargs):
            result = func(self, population, environment, *args, **kwargs)
            self.store_positions(population, kwargs['step'])
            return result
        return wrapper

    def store_positions(self, population, step):
        """
        Logs the position, and field of view of each agent in the population at the given step.
        """
        for i, agent in enumerate(population):
            self.trajectory_log[step, i] = [agent.position[0], agent.position[1], agent.direction, agent.perception_radius]

    @record_move
    def recycle_agents(self, population, environment, *args, **kwargs):
        """
        Make agents reusable for the next iteration.
        """
        for agent in population:
            if self.iteration != 0:
                agent.reset()
            agent.position = environment.get_random_position()

    @record_move
    def simulate_step(self, population, environment, *args, **kwargs):
        """
        Simulates agent action for one time step.
        Agents:
            - perceive the environment
            - choose an action
            - perform the action
        """
        for agent in population:
            direction_difference, food_proximity = agent.perceive(environment)
            direction, velocity_mode = agent.choose_action(direction_difference, food_proximity)
            agent.perform_action(environment, direction, velocity_mode)
    
    def store_consumed_meals(self, population):
        """
        Store the number of meals consumed by each agent in each iteration.
        """
        for i, agent in enumerate(population):
            self.meals_per_iteration[self.iteration, i] = agent.meals

    def evolve(self, population):
        """
        Exchange old population with a new population.
        """
        if not self.params.train_mode:
            return population
        evolutionary_algorithm = EvolutionaryAlgorithm(self.params)
        # sort population after performance in descending order
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        new_population = evolutionary_algorithm.evolve(sorted_population) 
        return new_population
    
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
        for key, value in kwargs.items():
            setattr(self, key, value)

        # one more for the initial positions
        self.simulation_steps = len(np.arange(0, self.total_time, self.delta_t)) + 1

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