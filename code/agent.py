import numpy as np
from tqdm import tqdm
from utils import Point, rectangle_from_points, inside_rectangle, intersect
import networkx as nx

class Agent:
    """
    An agent that navigates in an environment and eats food particles.
    Agents move with a constant velocity in the environment.
    The positioning of the agent in the environment is described by:
    - its 2D position
    - the direction it faces in world perspective (0 to 2pi).
    The agent eats food particles that are within its eat radius.

    All agents are subclasses of this class and follow a simple logic at each time step
    - perceive the environment
    - choose an action
    - perform the action.
    """

    def __init__(self, params):
        self.params = params
        self.eat_radius = params.eat_radius
        self.velocity = params.velocity
        self.direction = np.random.uniform(0, 2*np.pi)
        self.position = np.array(np.random.uniform(0, params.size, 2))
        # position of the agent at the last time step
        # used to check whether the agent stepped over food particles
        self.last_position = None
        # mask for food particles that have already been eaten
        self.food_mask = np.zeros(params.num_food, dtype=bool)
        # count how many food particles the agent has eaten
        self.meals = 0
        # whether the agent ate at the previous time step
        self.ate = False
        self.sensed_wall = False
    
    def perceive(self, environment):
        """
        Sense environment for food particles around current position.
        Consume food that is within the eat radius.

        Args:
            environment (Environment): the environment the agent navigates in            
        """
        self.ate = False
        _, food_distance, food_index = environment.get_closest_food(self)
        if food_distance and food_distance <= self.eat_radius:
            self.eat(food_index)
        self.check_path(environment)

    def check_path(self, environment):
        """
        Check whether on the path from the last position to the current position there was a food particle.
        
        Constructs a rectangle between the past and current position of the agent. Where
        - one side is two times the eat radius long and perpendicular to the direction of movement and
        - the other is the distance between the past and current position of the agent.
        If there is a yet undetected food particle inside the boundaries of the rectangle the agent must have traversed it.

        Args:
            environment (Environment): the environment the agent navigates in
        """
        # do not check path on first step of every iteration
        if self.last_position is None:
            return
        # do not check_path if boundary got crossed 
        if abs(self.last_position[0] - self.position[0]) >  environment.size / 2 or abs(self.last_position[1] - self.position[1]) > environment.size / 2:
            return
        point1 = Point(self.last_position[0], self.last_position[1])
        point2 = Point(self.position[0], self.position[1])
        hitbox = rectangle_from_points(point1, point2, self.eat_radius)
        for i, food_particle in enumerate(environment.food_positions):
            food_point = Point(food_particle[0], food_particle[1])
            if not self.food_mask[i] and inside_rectangle(hitbox, food_point):
                self.eat(i)
    
    def move(self, new_position, environment):
        """
        Move to new position and update last position.
        Don't move if agent would step over a wall.

        Args:
            new_position (np.array): new position of the agent
            environment (Environment): the environment the agent navigates in
        """
        for wall in environment.walls:
            if intersect(self.position, new_position, wall[0], wall[1]):
                self.sensed_wall = True
                return
        new_position = np.mod(new_position, environment.size)
        self.last_position = self.position
        self.position = new_position

    def eat(self, food_index):
        """
        Consume a food particle.

        Args:
            food_index (int): index of the food particle
        """
        self.food_mask[food_index] = True
        self.meals += 1
        self.ate = True

    def reset(self):
        """
        Reset the agent to its initial state.
        """
        self.food_mask = np.zeros(self.params.num_food, dtype=bool)
        self.meals = 0
        self.position = np.array(np.random.uniform(0, self.params.size, 2))
        self.direction = np.random.uniform(0, 2*np.pi)
        self.ate = False
        self.last_position = None
        self.sensed_wall = False


class LévyAgent(Agent):
    """
    A blind agent that navigates in an environment and follows Lévy walk like movement patterns.
    Movement:
    - agent chooses a random direction
    - agent chooses a step length according to a power law distribution
    - agent travels in the chosen direction for step length / velocity time steps
    """
    
    def __init__(self, params):
        super().__init__(params)
        # optimal Lévy exponent
        self.mu = params.mu
        self.pending_steps = 0
        self.step_length_log = []

    def choose_action(self, _):
        """
        Agent chooses a random direction and a step length according to a power law distribution.
        """
        if self.pending_steps == 0 or self.sensed_wall:
            self.sensed_wall = False
            self.direction = np.random.uniform(0, 2*np.pi)
            x = np.random.uniform(0, 1)
            step_length = int(1 / x**(1/(self.mu - 1)))
            self.pending_steps = step_length
            self.step_length_log.append(step_length)
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.

        Args:
            environment (Environment): the environment the agent navigates in
        """
        self.pending_steps -= 1
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

    def reset(self):
        super().reset()
        self.pending_steps = 0
        self.step_length_log = []

class ExponentialAgent(Agent):
    """ 
    A blind agent whose step lengths are drawn from an exponential distribution.
    Movement:
    - agent chooses a random direction
    - agent chooses a step length according to an exponential distribution
    - agent travels in the chosen direction for step length / velocity time steps
    """

    def __init__(self, params):
        super().__init__(params)
        self.alpha = params.alpha
        self.pending_steps = 0
        self.step_length_log = []

    def choose_action(self, _):
        """
        Agent chooses a random direction and a step length from an exponential distribution
        """
        if self.pending_steps == 0 or self.sensed_wall:
            self.sensed_wall = False
            self.direction = np.random.uniform(0, 2*np.pi)
            u = np.random.uniform(0, 1)
            step_length = int(-np.log(1 - u) / self.alpha) + 1
            self.pending_steps = step_length
            self.step_length_log.append(step_length)
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent

        Args:
            environment (Environment): the environment the agent navigates in
        """
        self.pending_steps -= 1
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

    def reset(self):
        super().reset()
        self.pending_steps = 0
        self.step_length_log = []

class BrownianAgent(Agent):
    """
    A blind agent that follows Brownian motion.
    Movement:
    - agent chooses a random direction
    - agent chooses a step length according to a normal distribution
    - agent travels in the chosen direction for step length / velocity time steps
    """

    def __init__(self, params):
        super().__init__(params)
        # parameters of the normal distribution
        self.mean = 0
        self.standard_deviation = 1
        self.pending_steps = 0
        self.step_length_log = []

    def choose_action(self, _):
        """
        Agent chooses a random direction and a step length according to a normal distribution.
        """
        if self.pending_steps == 0 or self.sensed_wall:
            self.sensed_wall = False
            self.direction = np.random.uniform(0, 2*np.pi)
            step_length = int(abs(np.random.normal(self.mean, self.standard_deviation))) + 1
            self.pending_steps = step_length
            self.step_length_log.append(step_length)
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.

        Args:
            environment (Environment): the environment the agent navigates in
        """
        self.pending_steps -= 1
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

    def reset(self):
        super().reset()
        self.pending_steps = 0
        self.step_length_log = []


class BallisticAgent(Agent):
    """
    A blind agent that moves straight and never turns.
    """

    def choose_action(self, _):
        """
        Never decides on any action. Keeps original direction.
        """
        pass
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.
        Args:
            environment (Environment): the environment the agent navigates in
        """
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

class ReservoirAgent(Agent):
    """
    A blind agent that is controlled by a neural reservoir.
    """

    def __init__(self, params, model = None):
        super().__init__(params)
        self.params = params
        if model is None:
            self.model = Reservoir(params.simulation_steps, params.num_neurons, params.burn_in_time, params.mean, params.standard_deviation)
        else:
            self.model = model
        self.time_step = 0
        self.output_log = []

    def choose_action(self, _):
        """
        Transforms the output of the reservoir to a direction.
        The tanh activation function squishes the output to the interval (-1, 1).
        It is then scaled by 2π, which means that the agent can make a full 2π turn clockwise or counterclockwise.
        That means each direction is covered twice, in the positive as well as in the negative spectrum and ultimatly means:
        - turn by 180° at -0.5 and 0.5
        - keep direction at 0, 1 and -1.
        """
        output = self.model.get_output(self.time_step)
        self.output_log.append(output)
        angle = output * 2 * np.pi
        direction = (self.direction + angle) % (2 * np.pi)
        self.direction = direction
        self.time_step += 1

    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.
        Args:
            environment (Environment): the environment the agent navigates in
            _ (None): unused
        """
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

    def reset(self):
        super().reset()
        self.time_step = 0
        self.output_log = []
        del self.model
        self.model = Reservoir(self.params.simulation_steps, self.params.num_neurons, self.params.burn_in_time, self.params.mean, self.params.standard_deviation)

class Reservoir():
    """
    A set of neurons that are randomly connected to each other.
    The neurons are randomly activated at the beginning and can be in two states: on or off.
    A neuron is activated if exactly one of its neighbors was active at the previous time step.
    """

    def __init__(self, time_steps, num_neurons=1000, burn_in_time=1000, mean=0, standard_deviation=0.032, use_small_world=False, k=10, p=0.1):
        """
        Args:
            time_steps (int): number of time steps to simulate
            num_neurons (int): number of neurons in the reservoir
            burn_in_time (int): number of time steps to simulate the reservoir activity before the actual simulation starts
            mean (float): mean of the normal distribution for the weights
            standard_deviation (float): standard_deviation of the normal distribution for the weights
        """
        self.time_steps = time_steps
        self.num_neurons = num_neurons
        self.burn_in_time = burn_in_time
        self.mean = mean
        self.standard_deviation = standard_deviation
        if use_small_world:
            G = nx.watts_strogatz_graph(num_neurons, k, p)
            adj_matrix = nx.to_numpy_array(G)
            weights = np.random.normal(self.mean, self.standard_deviation, (self.num_neurons, self.num_neurons))
            self.weight_matrix = weights * adj_matrix
        else:
            self.weight_matrix = np.random.normal(self.mean, self.standard_deviation, (self.num_neurons, self.num_neurons))
        self.output_weights = np.random.uniform(-1, 1, self.num_neurons)
        self.burn_in_state_matrix = self.burn_in()
        self.neuron_state_time_matrix = np.zeros((self.time_steps, self.num_neurons), dtype=float)
        self.run()

    def burn_in(self):
        """
        Kickstart, then let the reservoir run for burn_in_time steps.
        """
        burn_in_state_matrix = np.zeros((self.burn_in_time, self.num_neurons), dtype=float)
        # initialize neuron states from distribution
        burn_in_state_matrix[0] = np.tanh(np.random.normal(self.mean, self.standard_deviation, self.num_neurons))
        for t in range(1, self.burn_in_time):
            burn_in_state_matrix[t] = np.tanh(np.dot(self.weight_matrix, burn_in_state_matrix[t - 1]))
        return burn_in_state_matrix

    def run(self):
        """
        Simulate network activity over time. Start with the last state of the burn in period and let the reservoir run for time_steps.
        """
        self.neuron_state_time_matrix[0] = self.burn_in_state_matrix[-1]
        for t in range(1, self.time_steps):
            self.neuron_state_time_matrix[t] = np.tanh(np.dot(self.weight_matrix, self.neuron_state_time_matrix[t - 1]))

    def get_output(self, time_step):
        """
        Returns the output of the reservoir at the given time step.
        Multiplies reservoir state at the given time step with the output weights and applies tanh.
        -1 < output < 1
        """
        output = np.dot(self.neuron_state_time_matrix[time_step], self.output_weights)
        output = np.tanh(output)
        return output

if __name__ == "__main__":
    pass