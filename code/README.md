# Lévy Agent Simulation
Simulate foraging behaviour of simple agents controlled by a recurrent neural network model
in a 2D environment. \
Analyse under which circumstances the agents develop Lévy Walk like motion patterns.

# Project Set Up
Create a virtual environment, then install the required packages

    python3 -m venv levy-agent-simulation
    source levy-agent-simulation/bin/activate

# Parameters
Agent 
- input_size (int): number of input neurons (constant)
- hidden_size (int): number of hidden neurons 
- output_size (int): number of output neurons (constant)
- noise_neuron (bool): 1 if a noise neuron is used, 0 otherwise
- eat_radius (float): distance at which the agent eats a food particle
- slow_velocity (float): velocity of the agent in slow mode
- fast_velocity (float): velocity of the agent in fast mode
- slow_perception_radius (float): perception radius of the agent in slow mode
- fast_perception_radius (float): perception radius of the agent in fast mode
- num_food (int): number of food particles in the environment# slow_velocity = 1

Environment 
- size (int): size of the environment
- num_food (int): number of food particles
- border_buffer (float): minimum distance between food particles and the border of the environment
    ~10% of the environment size
- food_buffer (float): minimum distance between food particles# size = 200
    choose as max(fast_perception_radius, slow_perception_radius) + eat_radius so that agents cannot
    'see' next food item from current one 
- empty (bool): if true, no food particles are spawned into the environment

Evolution
- population_size (int): number of agents in the population
- elite_fraction (float): fraction of the best performing agents that will be copied to the next generation
- mutation_fraction (float): fraction of the population that will be mutated
- mutation_rate (float): probability of a weight being mutated
- mutation_strength (float): strength of the mutation
- tolerance (float): tolerance value to set weights
- train_mode (bool): whether evolution is enabled or not

Simulation
- total_time (int): total time of the simulation
- delta_t (int): time step increment (needed to translate from velocity units to position units)
- num_epochs (int): number of epochs
- iterations_per_epoch (int): number of iterations per epoch
- population (int): number of agents in the population
- load_simulation (bool): population is saved after the simulation is over and can be loaded by setting this to true

TODO only store the past simulation.

In each epoch the agents are evaluated iterations_per_epoch times.
And in each iteration the agent behaviour is simulated simulation_steps times

# Coding Conventions
Docstring format inspired by Google:

    """
    This is an example of Google style.

    Args:
        param1: This is the first param.
        param2: This is a second param.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """

Naming:
- function_name
- variable_name
- ClassName

Abbreviations:
- num: number
- params: parameters
- Rnn: Recurrent Neural Network

Structure Rules:
- modularization by files
- all parameters are collected in the parameter class
- each class extracts only the parameters that it needs itself

Notes:
Tried switching to gpu, but this actually seems to increase computation time.
Reasons for why gpu acceleration might not work:
- no batch processing
- small neural networks that are called very often
- device overhead is quite large 

BUGS:
- animation function is using multiprocessing, pay attention to modules that are not thread safe
  - tkinter for example
  - or matplotlib
- can cause errors like this:
  - "
    X connection to :1 broken (explicit kill or server shutdown).
        XIO:  fatal IO error 0 (Success) on X server ":1"
        after 183 requests (183 known processed) with 2 events remaining.
    "

#TODO
- agent model description
- ego and world perspective description
    clockwise is negative
    counter-clockwise is positive
    When the difference is exactly pi, it is always negative. 
    In ego perspective this does not matter, it is just because of the way the function is defined