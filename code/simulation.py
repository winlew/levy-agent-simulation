import numpy as np
from environment import Environment
from agent import *
from data_io import update_data, initialize_data, save_simulation_context, save_data
import config
import time

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
        self.iterations = params.iterations
        self.population_size = params.population_size
        self.iteration = 0
        # trajectory_log is used to store
        # - position
        # - direction
        # - whether agent consumed food
        # for every agent at every time step in a single iteration
        self.trajectory_log = np.zeros((self.params.simulation_steps, self.population_size, config.NUM_MOTION_ATTRIBUTES))
        self.data = initialize_data(self.params)

    def run(self, folder, population=None):
        """
        Run a simulation.

        Args:
            folder (str): store the simulation results here
            population (list): list of agents
        """
        environment = Environment(self.params)

        population = self.set_up_population(population)

        for _ in range(self.iterations):
            self.run_iteration(population, environment)
            self.iteration += 1
        
        # sum up consumed food particles over all time steps for every agent
        self.fitnesses = np.asarray(np.sum(self.data['ate'], axis=1))

        if self.params.save:
            save_simulation_context(folder, environment) 
            save_data(folder, self.data, population)

        return self.fitnesses

    def set_up_population(self, population):
        """
        Either use an existing population or create a population from scratch by instantiating
        agents with the agent class.

        Args:
            population (list): list of agents
        """
        if population:
            return population  
        return [self.agent(self.params) for _ in range(self.population_size)]
    
    def record_iteration_data(func):
        """
        Decorator to update the data with one iteration.
        """
        def wrapper(self, population, environment):
            result = func(self, population, environment)
            update_data(self.data, self.iteration, self.trajectory_log)
            return result
        return wrapper

    def record_move(func):
        """
        Decorator to update the current iteration with the data from one step.
        Requires the wrapped function to be passed the step number.
        """
        def wrapper(self, population, environment, *args, **kwargs):
            result = func(self, population, environment, *args, **kwargs)
            self.collect_data(population, kwargs['step'])
            return result
        return wrapper
    
    @record_iteration_data
    def run_iteration(self, population, environment):
        """
        Executes a singe iteration that simulates agent foraging for a certain time.

        Args:
            population (list): list of agents
            environment (Environment): the environment the agents navigate in
        """
        self.recycle_agents(population, environment, step = 0)
        for step in range(self.params.simulation_steps - 1):
            self.simulate_step(population, environment, step = step + 1)
        return population

    @record_move
    def recycle_agents(self, population, environment, *args, **kwargs):
        """
        Reset agents so they can be reused in the next iteration.

        Args:
            population (list): list of agents
            environment (Environment): the environment the agents navigate in
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

        Args:
            population (list): list of agents
            environment (Environment): the environment the agents navigate in
        """
        for agent in population:
            perception = agent.perceive(environment)
            agent.choose_action(perception)
            agent.perform_action(environment)

    def collect_data(self, population, step):
        """
        Logs position, direction and whether the agent consumed food for each agent at the given step.

        Args:
            population (list): list of agents
            step (int): current time step
        """
        for i, agent in enumerate(population):
            self.trajectory_log[step, i] = [agent.position[0], agent.position[1], agent.direction, False]
            # this function is called after the agent has moved
            # so the ate value has to be stored for the previous step
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

if __name__ == '__main__':
    from data_io import extract_gif_frames
    extract_gif_frames('001500_10_extracted', 'animation_1.gif')
    pass