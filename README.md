# Lévy Agent Simulation
Simulate foraging behaviour of simple autonomous agents in a 2D environment. \
Analyse under which circumstances the agents develop Lévy walk like motion patterns.

<img src="./resources/simulation_example.gif" alt="simulation_example" width="500"/>

## Set Up
Clone the repository

```git clone git@github.com:winlew/levy-agent-simulation.git```

Create a virtual environment 

```python3 -m venv levy-agent-simulation```\
```source levy-agent-simulation/bin/activate```

Install the required packages

```pip install -r requirements.txt```

Run a simulation by executing main.py.
Configure it by modifying parameters.json.

## Environment
The environment is the 2D area in which the simulation takes place.
- quadratic
- has periodic boundaries
- contains randomly distributed food particles.

## Agents
The positioning of the agent in the environment is described by
- its 2D position
- the direction it faces.

There are multiple agent types:
- Ballistic agent (walks straight only)
- Lévy agent (follow Lévy walk like motion patterns)
- Brownian agent (follows brownian motion)
- RNN agent (behavior is controlled by a hybrid 3 layer neural network)
- Reservoir agent (controlled by reservoir computing)

## Training Through Evolution
Not all agents can be trained. 
Each epoch the next generation is assembled from
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
- type (class): class of the agent that is used one of {'rnn', 'levy', 'ballistic', 'brownian', 'reservoir'}
- eat_radius (float): distance at which the agent eats a food particle
- velocity (float): velocity of the agent 
- perception_radius (float): perception radius of the agent

Environment 
- size (int): size of the environment
- num_food (int): number of food particles in the environment
- border_buffer (float): minimum distance between food particles and the border of the environment
- food_buffer (float): minimum distance between food particles choose as perception_radius + eat_radius so that agents cannot 'see' next food item from current one 
- empty (bool): if true, no food particles are spawned into the environment
- resetting_boundary (bool): if true, then there are walls around the edges of the environment that the agents cannot cross

Evolution
- population_size (int): number of agents in the population
- elite_fraction (float): fraction of the best performing agents that will be copied to the next generation
- mutation_fraction (float): fraction of the population that will be mutated
- mutation_rate (float): probability of a weight being mutated
- mutation_strength (float): amplitude of the mutation
- tolerance (float): tolerance value to set weights

Simulation
- total_time (int): total time of the simulation (cannot be changed after Params object has been created)
- delta_t (int): time step increment (needed to translate from velocity units to position units)
- num_epochs (int): number of epochs
- iterations_per_epoch (int): number of iterations per epoch

Settings
- intervall_save (int): determines the frequency of saves (< num_epochs)

### Perspectives
There are two types of perspectives used in the project
- ego perspective $[-\pi, \pi]$
- world perspective $[0, 2\pi]$

Agents themselves perceive the world in ego perspective while their actual orientation is in world perspective.
That allows the agents to perceive the world relative to their own orientation where 0 is the current direction they are facing.
In world perspective 0 is 3 o'clock.

Note: For ego perspective clockwise is negative and counter-clockwise is positive.

# Limitations
- agents can eat at max 1 food particle at each time step
- food particles have to have a minimal distance that is larger than perception_radius from each other
- check_path() is not periodic boundary safe but does not have to be either if border_buffer > velocity

# Coding Conventions
Standards to ensure consistency in the project. 
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
