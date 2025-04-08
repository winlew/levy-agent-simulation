import numpy as np
import torch
import torch.nn as nn
from utils import Point, rectangle_from_points, inside_rectangle, vector_to_angle, calculate_angle_difference, intersect

class Agent:
    """
    An agent that navigates in an environment and eats food particles.
    Agents move with a constant velocity in the environment.
    The positioning of the agent in the environment is described by:
    - its 2D position
    - the direction it faces in world perspective (0 to 2pi)
    The perception radius determines how far the agent can sense food particles.
    The agent eats food particles that are within its eat radius.

    All agents are subclasses of this class and follow a simple logic at each time step
    - perceive the environment
    - choose an action
    - perform the action.
    """

    def __init__(self, params):
        self.params = params
        self.eat_radius = params.eat_radius
        self.perception_radius = params.perception_radius
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
        self.step = 0

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
        # do not check path on first step
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
        self.step += 1
        for wall in environment.walls:
            if intersect(self.position, new_position, wall[0], wall[1]):
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
        # agent is blind, so it only senses food particles that are within its body 
        self.perception_radius = params.eat_radius
        # optimal Lévy exponent
        self.mu = 1 
        self.pending_steps = 0

    def choose_action(self, _):
        """
        Agent chooses a random direction and a step length according to a power law distribution.
        """
        if self.pending_steps == 0:
            self.direction = np.random.uniform(0, 2*np.pi)
            x = np.random.uniform(0, 1)
            step_length = int(1 / x**(1/self.mu))
            self.pending_steps = step_length
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.

        Args:
            environment (Environment): the environment the agent navigates in
            _ (None): unused
        """
        self.pending_steps -= 1
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

class BrownianAgent(Agent):
    """
    A blind agent that follows Brownian motion.
    """

    def __init__(self, params):
        super().__init__(params)
        # agent is blind, so it only senses food particles that are within its body 
        self.perception_radius = params.eat_radius
        # parameters of the normal distribution
        self.mean = 0
        self.variance = 1
        self.pending_steps = 0

    def choose_action(self, _):
        """
        Agent chooses a random direction and a step length according to a normal distribution.
        """
        if self.pending_steps == 0:
            self.direction = np.random.uniform(0, 2*np.pi)
            step_length = int(abs(np.random.normal(self.mean, self.variance))) + 1
            self.pending_steps = step_length
    
    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.

        Args:
            environment (Environment): the environment the agent navigates in
            _ (None): unused
        """
        self.pending_steps -= 1
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

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
            _ (None): unused
        """
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

class RnnAgent(Agent):
    """
    Agent that is controlled by a Rnn.
    """

    def __init__(self, params, model=None):
        super().__init__(params)
        self.model = model if model else Rnn(params)
        self.hidden_state = torch.zeros(1, self.model.hidden_size)

    def perceive(self, environment):
        """
        Returns:
            food_direction (float): direction to the closest food particle in radians (ego perspective)
            food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise
        """
        self.ate = False
        food_position, food_distance, food_index = environment.get_closest_food(self)
        if food_distance is None:
            direction_difference = 0
            food_proximity = 0
        elif food_distance < self.eat_radius:
            self.eat(food_index)
            direction_difference = 0
            food_proximity = 0
        else:
            food_direction_vector = food_position - self.position
            unit_food_direction_vector = food_direction_vector / np.linalg.norm(food_direction_vector)
            food_direction = vector_to_angle(unit_food_direction_vector)
            direction_difference = calculate_angle_difference(food_direction, self.direction)
            food_proximity = 1
        self.check_path(environment)
        perception = (direction_difference, food_proximity)
        return perception

    def choose_action(self, perception): 
        """
        Given the position of the closest food particle, agent decides where and how fast to go.

        After perceiving the environment, the agent decides on its next action.
        To do so, the neural network is fed with the following sensory information:
        - food_direction (float): direction to the closest food particle in radians (ego perspective)
        - food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise
        - noise (float): random noise to add to the decision (if noise_neuron is True)
        The hidden state of the recurrent layer is updated and reused for the next decision.
        It outputs:
        - steering command (int): 0 for keep direction, 1 for turn
        - angle to turn (float): angle in ego perspective to turn in radians

        Args:
            perception (tuple)
                food_direction (float): direction to the closest food particle in radians (ego perspective)
                food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise

        Returns:
            direction (float): angle in world perspective (where the agent turns next)
        """
        direction_difference = perception[0]
        food_proximity = perception[1]
        if self.model.noise_neuron:
            noise = np.random.randn() * (1 - food_proximity) # noise only if food is not in promixity
            input_data = torch.Tensor([[direction_difference, food_proximity, noise]])
        else:
            input_data = torch.Tensor([[direction_difference, food_proximity]])

        output, self.hidden_state = self.model(input_data, self.hidden_state)

        # binary step function
        steering_command = 0 if output[0] < 0 else 1
        # scale tanh() with pi to choose direction in ego perspective
        angle = torch.tanh(output[1]).numpy() * np.pi
        # transform angle to world perspective
        direction = (self.direction + steering_command * angle) % (2 * np.pi)
        self.direction = direction

    def perform_action(self, environment):
        """
        Calculate the next position and move the agent.

        Args:
            environment (Environment): the environment the agent navigates in
            direction (float): angle in world perspective (where the agent turns next)
        """
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        self.move(new_position, environment)

    def reset(self):
        """
        Reset the agent to its initial state.
        """
        self.food_mask = np.zeros(self.params.num_food, dtype=bool)
        self.meals = 0
        self.position = np.array(np.random.uniform(0, self.params.size, 2))
        self.direction = np.random.uniform(0, 2*np.pi)
        self.hidden_state = torch.zeros(1, self.model.hidden_size)
        self.ate = False

class ReservoirAgent(Agent):
    """
    A blind agent that is controlled by an Echo State Network (ESN).
    """

    def __init__(self, params, model = None):
        super().__init__(params)
        self.params = params
        if model is None:
            self.model = Reservoir(params.simulation_steps)
            self.model.run()
        else:
            self.model = model
        self.time_step = 0

    def choose_action(self, _):
        """
        Scale the direction with the output of the neural reservoir.
        """
        output = self.model.get_output(self.time_step)
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
        """
        Reset the agent to its initial state.
        """
        self.food_mask = np.zeros(self.params.num_food, dtype=bool)
        self.meals = 0
        self.position = np.array(np.random.uniform(0, self.params.size, 2))
        self.direction = np.random.uniform(0, 2*np.pi)
        self.time_step = 0
        self.ate = False

class Rnn(nn.Module):
    """
    A hybrid 3 layer neural network
    - input layer
        - direction to food particle in ego perspective (-pi to pi)
        - proximity to food particle (0: out of perception radius, 1: within perception radius)
        - optional noise neuron
    - recurrent hidden layer
    - fully connected output layer
        - steering command (0: keep direction, 1: turn)
        - angle to turn in ego perspective (-pi to pi)

    The network is not trainable, because it is intended to be trained by an evolutionary algorithm across generations.
    """

    def __init__(self, params, noise_neuron=True, hidden_size=4):
        """
        Args:
            noise_neuron (bool): whether to include additional noise as input
            hidden_size (int): how many hidden neurons
        """
        super(Rnn, self).__init__()
        self.params = params
        self.input_size = 2
        self.noise_neuron = noise_neuron
        if self.noise_neuron:
            self.input_size += 1
        self.hidden_size = hidden_size
        self.output_size = 2 
        self.Rnn = nn.RNN(self.input_size, self.hidden_size, num_layers = 1)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # no training is required
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, input_data, hidden_state):
        """
        Mapping sensory information to motor controls.
        The input data is a sequence of perceptions the agent made over time.
        This sequence usually has length 1, as the agent 'thinks' at each time step.
        The recurrent layer keeps track of an internal hidden state that reflects processed information 
        of the past time steps and is reused for the current output of the hidden layer.
        The output layer generates motor control commands.

        Args:
            input_data (torch.Tensor(sequence_length, input_size)): sequence of perceptions over time
            hidden_state (torch.Tensor(1, hidden_size)): hidden_state of last pass through the network

        Returns:
            output (torch.Tensor(output_size)): motor controls of agent
            hidden_state (torch.Tensor(1, hidden_size)): hidden_state of current pass through the network
        """

        # hidden_output has shape (sequence_length, hidden_size)
        # hidden_state is the new internal memory of the model at the current time step 
        hidden_output, hidden_state = self.Rnn(input_data, hidden_state)
        # if input data contains perceptions over multiple time steps only the last one 
        # is of interest as it contains processed information from all previous time steps
        output = self.fc(hidden_output[-1, :])
        return output, hidden_state

class Reservoir():
    """
    A set of neurons that are randomly connected to each other.
    The neurons are randomly activated at the beginning and can be in two states: on or off.
    A neuron is activated if exactly one of its neighbors was active at the previous time step.
    """

    def __init__(self, time_steps, num_neurons=300, connection_probability=0.01, initially_active_ratio=0.01):
        """
        Args:
            num_neurons (int): number of neurons in the reservoir
            connection_probability (float): probability of a connection between two neurons
            initially_active_ratio (float): ratio of initially active neurons
        """
        self.time_steps = time_steps
        self.num_neurons = num_neurons
        self.connection_probability = connection_probability
        self.initially_active_ratio = initially_active_ratio
        self.connection_matrix = np.random.random((num_neurons, num_neurons)) < connection_probability
        self.neuron_state_time_matrix = np.zeros((time_steps, num_neurons), dtype=int)
        # randomly activate 10% of neurons initially
        self.neuron_state_time_matrix[0] = (np.random.random(num_neurons) < initially_active_ratio).astype(int)

        # weights connecting each neuron to the output node
        self.weights = np.random.uniform(-1, 1, num_neurons)

    def run(self):
        """
        Simulate network activity over time.
        """
        for t in range(1, self.time_steps):
            # for each neuron, check if any of its connected neighbors were active at t-1
            for i in range(self.num_neurons):
                # get all neurons that connect to neuron i
                neighbors = np.where(self.connection_matrix[:, i])[0]
                # check if any neighbors were active at previous time step
                if np.sum(self.neuron_state_time_matrix[t - 1, neighbors] == 1) == 1:
                    self.neuron_state_time_matrix[t, i] = 1

    def get_output(self, time_step):
        """
        Multiply the state of the reservoir at the given time_step with its weight.
        Apply the sigmoid function to the weighted sum.
        """
        output = np.dot(self.neuron_state_time_matrix[time_step], self.weights)
        output = 1 / (1 + np.exp(-output))
        return output