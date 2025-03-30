# Lévy Agent Simulation
Simulate foraging behaviour of simple autonomous agents in a 2D environment. \
Analyse under which circumstances the agents develop Lévy Walk like motion patterns.

![simulation_example.gif](./resources/simulation_example.gif)

## Set Up
Create a folder for the project

```mkdir levy-agent-simulation```

Clone the github repository

```git clone git@github.com:winlew/levy-agent-simulation.git```

Create a virtual environment 

```python3 -m venv levy-agent-simulation```\
```source levy-agent-simulation/bin/activate```

Install the required packages

```pip install -r requirements.txt```

Run a simulation by executing main.py.

## Environment
The environment is the area in which the simulation takes place.
- quadratic
- has periodic boundaries
- contains randomly distributed food particles

## Agent Architecture and Model
The positioning of the agent in the environment is described by
- its 2D position
- the direction it faces

The behavior of the agent is controlled by a hybrid 3 layer neural network
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

## Training
Agents are trained by an evolutionary algorithm. Each epoch the next generation is assembled from
- elites
  - best performing fraction of population
  - small weights are set to zero
- mutated
  - random chosen agent that is mutated
  - small weights are set to zero
- crossovers
  - between elites and mutated

## Simulation
Each simulation runs for a certain number of epochs.
In each epoch the performance of each agent is evaluated for a designated number of iterations where each iteration simulates agent behavior for a limited number of time steps.
At the end of each epoch the current population is exchanged with the next.
Simulations are configured by the parameters.json file. Here a detailed explanation of the purpose of each parameter:

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

Environment 
- size (int): size of the environment
- num_food (int): number of food particles in the environment
- border_buffer (float): minimum distance between food particles and the border of the environment
- food_buffer (float): minimum distance between food particles choose as max(fast_perception_radius, slow_perception_radius) + eat_radius so that agents cannot 'see' next food item from current one 
- empty (bool): if true, no food particles are spawned into the environment

Evolution
- population_size (int): number of agents in the population
- elite_fraction (float): fraction of the best performing agents that will be copied to the next generation
- mutation_fraction (float): fraction of the population that will be mutated
- mutation_rate (float): probability of a weight being mutated
- mutation_strength (float): amplitude of the mutation
- tolerance (float): tolerance value to set weights
- train_mode (bool): whether evolution is enabled or not

Simulation
- total_time (int): total time of the simulation
- delta_t (int): time step increment (needed to translate from velocity units to position units)
- num_epochs (int): number of epochs
- iterations_per_epoch (int): number of iterations per epoch
- population (int): number of agents in the population
- load_simulation (bool): population is saved after the simulation is over and can be loaded by setting this to true

### Perspectives
There are two types of perspectives used in the project
- ego perspective $[-\pi, \pi]$
- world perspective $[0, 2\pi]$
  
Agents view the world in ego perspective, why they are placed inside the environment by using world perspective.
For ego perspective:
- clockwise is negative and
- counter-clockwise is positive

# Coding Standards
Conventions to ensure consistency in the project. 
Regarding the structure I tried to modularize by files. All parameters are collected in the parameter class but each class only extracts the parameters it needs.
- Docstring format inspired by Google:

```
    """
    This is an example.

    Args:
        param1 (type): This is the first param.
        param2 (type): This is a second param.

    Returns:
        value (type): This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """
```

- Naming:
  - function_name
  - variable_name
  - ClassName

- Abbreviations:
  - num: number
  - params: parameters
  - Rnn: Recurrent Neural Network