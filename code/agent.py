import numpy as np
import torch
import torch.nn as nn
import random
import math

class Rnn(nn.Module):
    """
    A hybrid 3 layer neural network
    - input layer
        - velocity mode (0: slow, 1: fast)
        - direction to food particle in ego perspective (-pi to pi)
        - proximity to food particle (0: out of perception radius, 1: within perception radius)
        - optional noise neuron
    - recurrent hidden layer
    - fully connected output layer
        - velocity mode (0: slow, 1: fast)
        - steering command (0: keep direction, 1: turn)
        - angle to turn in ego perspective (-pi to pi)

    The network is not trainable, because it is intended to be trained by an evolutionary algorithm across generations.
    """

    def __init__(self, params):
        """
        Args:
            params (Params): class that stores parameters for the simulation
                input_size (int): number of input neurons
                hidden_size (int): number of hidden neurons
                output_size (int): number of output neurons
                noise_neuron (bool): 1 if a noise neuron is used, 0 otherwise
        """
        super(Rnn, self).__init__()
        self.params = params
        self.input_size = params.input_size
        self.noise_neuron = params.noise_neuron
        if self.noise_neuron:
            self.input_size += 1
        self.hidden_size = params.hidden_size
        self.output_size = params.output_size
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


# DISFUNCTIONAL ATM
class Agent:
    """
    An agent that navigates in an environment and eats food particles.

    The behaviour of the agent is controlled by its "brain", the Rnn.

    The positioning of the agent in the environment is described by:
    - its 2D position
    - the direction it faces in world perspective (0 to 2pi)
    The agent can be in two different velocity modes:
    - slow
    - fast
    The perception radius determines how far the agent can sense food particles and is dependent
    on the velocity mode of the agent.
    The agent eats food particles that are within its eat radius.
    """

    def __init__(self, model, params):
        """
        Args:
            model (Rnn): the "brain" that controls the agent
            params (Params): class that stores parameters for the simulation
                eat_radius (float): distance at which the agent eats a food particle
                slow_velocity (float): velocity of the agent in slow mode
                fast_velocity (float): velocity of the agent in fast mode
                slow_perception_radius (float): perception radius of the agent in slow mode
                fast_perception_radius (float): perception radius of the agent in fast mode
                num_food (int): number of food particles in the environment
        """
        self.model = model
        self.params = params
        self.eat_radius = params.eat_radius
        # 0: slow, 1: fast
        self.velocity_mode = 0
        # two different speeds to choose from
        self.slow_velocity = params.slow_velocity
        self.fast_velocity = params.fast_velocity
        # two different perception radii based on current velocity mode
        self.slow_perception_radius = params.slow_perception_radius
        self.fast_perception_radius = params.fast_perception_radius
        # initialize in slow mode
        self.velocity = self.slow_velocity
        self.perception_radius = self.slow_perception_radius
        # hidden state of the recurrent layer is empty at the beginning
        self.hidden_state = torch.zeros(1, self.model.hidden_size)
        # random direction in world coordinates
        self.direction = np.random.uniform(0, 2*np.pi)
        self.position = np.array([0,0])
        # hide food particles that have been eaten
        self.food_mask = np.zeros(params.num_food, dtype=bool)
        # number of food particles eaten
        self.meals = 0

    def perceive(self, environment):
        """
        The agent senses the environment for the closest food particle.
        If it is within the agents eat radius, it is consumend immediatly.

        Args:
            environment (Environment): the environment the agent navigates in

        Returns:
            food_direction (float): direction to the closest food particle in radians (ego perspective)
            food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise
        """

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

        return direction_difference, food_proximity
    
    def choose_action(self, direction_difference, food_proximity):
        """
        Given the position of the closest food particle, agent decides where and how fast to go.

        After perceiving the environment, the agent decides on its next action.
        To do so, the neural network is fed with the following sensory information:
        - velocity_mode (int): 0 for slow, 1 for fast
        - food_direction (float): direction to the closest food particle in radians (ego perspective)
        - food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise
        - noise (float): random noise to add to the decision (if noise_neuron is True)
        The hidden state of the recurrent layer is updated and reused for the next decision.
        It outputs:
        - velocity_mode (int): 0 for slow, 1 for fast
        - steering command (int): 0 for keep direction, 1 for turn
        - angle to turn (float): angle in ego perspective to turn in radians

        Args:
            food_direction (float): direction to the closest food particle in radians (ego perspective)
            food_proximity (bool): 1 if the food particle is within the perception radius, 0 otherwise

        Returns:
            direction (float): angle in world perspective (where the agent turns next)
            velocity_mode (int): 0 for slow, 1 for fast
        """

        if self.model.noise_neuron:
            noise = np.random.randn() * (1 - food_proximity) # noise only if food is not in promixity
            input_data = torch.Tensor([[self.velocity_mode, direction_difference, food_proximity, noise]])
        else:
            input_data = torch.Tensor([[self.velocity_mode, direction_difference, food_proximity]])

        output, self.hidden_state = self.model(input_data, self.hidden_state)

        # binary step functions
        velocity_mode = 0 if output[0] < 0 else 1
        steering_command = 0 if output[1] < 0 else 1
        # scale tanh() with pi to choose direction in ego perspective
        angle = torch.tanh(output[2]).numpy() * np.pi
        # transform angle to world perspective
        direction = (self.direction + steering_command * angle) % (2 * np.pi)
        return direction, velocity_mode

    def perform_action(self, environment, direction, velocity_mode):
        """
        Agent moves in the environment according to its decision.

        Args:
            environment (Environment): the environment the agent navigates in
            direction (float): direction in world coordinates of where the agent decided to go
            velocity_mode (int): 0 for slow, 1 for fast
        """

        self.direction = direction
        self.velocity_mode = 0 # velocity_mode TODO
        if self.velocity_mode == 0:
            self.velocity = self.slow_velocity
            self.perception_radius = self.slow_perception_radius
        else:
            self.velocity = self.fast_velocity
            self.perception_radius = self.fast_perception_radius
        
        new_position = self.position + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.velocity * self.params.delta_t
        new_position = np.mod(new_position, environment.size)

        legal_move = True
        for wall in environment.walls:
            if intersect(self.position, new_position, wall[0], wall[1]):
                legal_move = False
                break
        if legal_move:
            self.position = new_position

        # if not intersect(self.position, new_position, environment.wall[0], environment.wall[1]):
        #     self.position = new_position
        #     # correct for periodic boundary conditions
        #     self.position = np.mod(self.position, environment.size)

        # wrap around periodic boundary conditions
        # self.position = np.mod(new_position, environment.size)

    def eat(self, food_index):
        """
        Agent consumes a food particle.

        Args:
            food_index (int): index of the food particle
        """
        self.food_mask[food_index] = True
        self.meals += 1

    def reset(self):
        """
        Reset the agent to its initial state.
        The agents position is taken care of by the environment.
        """
        self.hidden_state = torch.zeros(1, self.model.hidden_size)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.velocity_mode = 0
        self.velocity = self.slow_velocity
        self.perception_radius = self.slow_perception_radius
        self.food_mask = np.zeros(self.params.num_food, dtype=bool)
        self.meals = 0


class LévyAgent:
    """
    A blind agent that navigates in an environment follows a Lévy Walk like movement pattern and eats food particles.
    The positioning of the agent in the environment is described by:
    - its 2D position
    - the direction it faces in world perspective (0 to 2pi)

    Movement:
    - agent chooses a random direction
    - agent chooses a step length according to a power law distribution
    - agent travels in the chosen direction for step length / velocity time steps
    """
    
    def __init__(self, params, velocity):
        self.eat_radius = params.eat_radius
        # agent is blind, so it only senses food particles that are within its body 
        self.perception_radius = params.eat_radius
        self.velocity = velocity
        self.direction = np.random.uniform(0, 2*np.pi)
        self.position = np.array([0,0])
        self.last_position = None
        self.food_mask = np.zeros(params.num_food, dtype=bool)
        # optimal Lévy exponent
        self.mu = 2 # but for destructable targets mu=1
        self.meals = 0
        self.pending_steps = 0
        self.num_food = params.num_food
        # internal count of the simulation steps
        self.step = 0
        self.meal_timeline = np.zeros(params.simulation_steps)
    
    def perceive(self, environment):
        """
        Agent senses the environment for the closest food particle.
        If it is within the agents eat radius, it is consumed immediatly.

        Args:
            environment (Environment): the environment the agent navigates in            
        """
        _, food_distance, food_index = environment.get_closest_food(self)
        if food_distance and food_distance <= self.eat_radius:
            self.eat(food_index)
        self.check_path(environment)

    def choose_action(self):
        """
        Agent chooses a random direction and a step length according to a power law distribution.
        """
        self.direction = np.random.uniform(0, 2*np.pi)
        x = np.random.uniform(0, 1)
        step_length = int(1 / x**(1/self.mu))
        self.pending_steps = step_length

    def check_path(self, environment):
        """
        Used to check whether agent stepped over food particle after updating its position.
        
        Constructs a rectangle between the past and current position of the agent. Where
        - side A is two times the eat radius long and perpendicular to the direction of movement and
        - side B is the distance between the past and current position of the agent.
        If there is a yet undetected food particle inside the boundaries of the rectangle, the agent consumes it.

        Args:
            environment (Environment): the environment the agent navigates in
        """
        # do not check path on first step
        if self.last_position is None:
            return
        # TODO check path is not periodic boundary safe!
        # TODO this method is domain size dependent
        # if a periodic boundary got crossed, disable check path for now
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

        Args:
            new_position (np.array): new position of the agent
            environment (Environment): 2D environment the agent navigates in
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
        self.meal_timeline[self.step] += 1

    def reset(self):
        """
        Reset the agent to its initial state.
        The agents position is taken care of by the environment.
        """
        self.food_mask = np.zeros(self.num_food, dtype=bool)
        self.meals = 0

class BallisticAgent:
    """
    Blind agent that moves straight and never turns.
    """

    def __init__(self, params, velocity):
        self.eat_radius = params.eat_radius
        self.perception_radius = params.eat_radius
        self.velocity = velocity
        self.direction = np.random.uniform(0, 2*np.pi)
        self.position = np.array([0,0])
        self.last_position = None
        self.food_mask = np.zeros(params.num_food, dtype=bool)
        self.meals = 0
        self.num_food = params.num_food
        # internal count of the simulation steps
        self.step = 0
        self.meal_timeline = np.zeros(params.simulation_steps)
    
    def perceive(self, environment):
        """
        Sense environment for food particles around current position.
        Consume food that is within the eat radius.

        Args:
            environment (Environment): the environment the agent navigates in            
        """
        _, food_distance, food_index = environment.get_closest_food(self)
        if food_distance and food_distance <= self.eat_radius:
            self.eat(food_index)
        self.check_path(environment)

    def check_path(self, environment):
        """
        Used to check whether agent stepped over food particle after updating its position.
        
        Constructs a rectangle between the past and current position of the agent. Where
        - side A is two times the eat radius long and perpendicular to the direction of movement and
        - side B is the distance between the past and current position of the agent.
        If there is a yet undetected food particle inside the boundaries of the rectangle, the agent consumes it.

        Args:
            environment (Environment): the environment the agent navigates in
        """
        # do not check path on first step
        if self.last_position is None:
            return
        # TODO check path is not periodic boundary safe!
        # TODO this method is domain size dependent
        # if a periodic boundary got crossed, disable check path for now
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
        Move agent to new position and update last position.

        Args:
            new_position (np.array): new position of the agent
            environment (Environment): 2D environment the agent navigates in
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
        Agent consumes a food particle.

        Args:
            food_index (int): index of the food particle
        """
        self.food_mask[food_index] = True
        self.meals += 1
        self.meal_timeline[self.step] += 1

    def reset(self):
        """
        Reset the agent to its initial state.
        The agents position is taken care of by the environment.
        """
        self.food_mask = np.zeros(self.num_food, dtype=bool)
        self.meals = 0


def vector_to_angle(normalized_vector):
    """
    Transforms a normalized 2D vector to an angle in radians.
    
    Args:
        normalized_vector (np.array): 2D vector with length 1

    Returns:
        angle (float): angle in radians (ego perspective)
    """
    angle = np.arctan2(normalized_vector[1], normalized_vector[0])
    return angle

def calculate_angle_difference(food_direction, agent_direction):
    """
    Calculates the difference between two angles.
    1. Shift food direction to world perspective by adding pi
    2. Substract the angles
    3. Transform negative differences to positive ones by wrapping around 2pi
    4. Shift back to ego perspective by substracting pi

    Args:
        food_direction (float): angle from agent to food particle in radians (ego perspective)
        agent_direction (float): angle that the agent is facing in radians (world perspective)

    Returns:
        delta (float): angle difference in radians (ego perspective)
    """
    assert(-np.pi <= food_direction <= np.pi)
    assert(0 <= agent_direction <= 2*np.pi)
    delta = (food_direction + np.pi - agent_direction) % (2 * np.pi) - np.pi
    return delta

# TODO refactor and understand
# this logic does not work for periodic boundaries
def counter_clockwise(p1, p2, p3):
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return counter_clockwise(A,C,D) != counter_clockwise(B,C,D) and counter_clockwise(A,B,C) != counter_clockwise(A,B,D)


class Point:
    """
    A point in 2D space.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        return Point(self.x * other, self.y * other)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y

class Rectangle:
    """
    Rectangle in 2D space. Defined by its four corner points.
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

def rectangle_from_points(point1, point2, radius):
    """
    Constructs a rectangle object that spans between the two given points.
    The vector between the two points is the midline of the rectangle.
    The sides perpendicular to the midline have length two times the radius. 
    """
    midline = point2 - point1
    perpendicular_vector = Point(-midline.y / math.sqrt(midline.dot(midline)), midline.x / math.sqrt(midline.dot(midline)))
    a = point1 + perpendicular_vector * radius
    b = point1 - perpendicular_vector * radius
    c = point2 - perpendicular_vector * radius
    d = point2 + perpendicular_vector * radius
    return Rectangle(a, b, c, d)

def inside_rectangle(rectangle, point):
    """
    Tests if a point is inside the borders of a rectangle.
    Chooses any corner point as the reference point.
    Then checks whether the projections of the vector from this reference point to the point and the sides of the rectangle are within the borders.
    """
    point_of_reference = rectangle.a
    rectangle_width_vector = rectangle.b - point_of_reference
    rectangle_height_vector = rectangle.d - point_of_reference
    point_difference_vector = point - point_of_reference
    relative_width_projection = rectangle_width_vector.dot(point_difference_vector) / rectangle_width_vector.dot(rectangle_width_vector)
    relative_height_projection = rectangle_height_vector.dot(point_difference_vector) / rectangle_height_vector.dot(rectangle_height_vector)
    if 0 <= relative_width_projection <= 1 and 0 <= relative_height_projection <= 1:
        return True 
    else:
        return False

if __name__ == '__main__':
    pass