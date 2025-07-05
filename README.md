# Lévy Agent Simulation
Lévy walks are considered as the optimal search strategy when blind agents explore sparse environments with unknown target distribution.

<p align="center">
  <img src="./resources/simulation_example.gif" alt="simulation_example" width="500"/>
</p>

# Objective
The goal of this project is to analyse under which conditions Lévy walk like motion patterns appear in agents that forage in a simple 2D world.
We aim to investigate whether agents controlled by neural reservoirs can exhibit Lévy walk like behavior, and how their fitness compares to agents with other control mechanisms.

## Set Up
Clone the repository

```git clone git@github.com:winlew/levy-agent-simulation.git```

Create a virtual environment 

```python3 -m venv levy-agent-simulation```\
```source levy-agent-simulation/bin/activate```

Install the required packages

```pip install -r requirements.txt```

Configure a simulation in parameters.json. Then start it by executing main.py. 
It will prompt you in the terminal to specify the location on where to store the results inside the data folder.

# Simulation
In each iteration a population of independent agents searches the environment for targets (food particles) for a limited number of time steps.
The fitness of each agent is measured by the amount of food particles that it consumed.

## Environment
2D quasi-continuous plane containing food particles, agents and optionally walls segments.

Food Particles:
- spawn in randomly at the beginning of the simulation with the restrictions:
  - cannot be closer together than a certain threshold
  - have a minimum margin towards the boundaries
- destructible
- increase agent fitness by 1 if consumed

Boundaries:
| Boundary Condition | Explanation                                                     | Consideration                                                                                                     |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| periodic           | crossing the boundary makes agent reappear at the opposite side | beneficial for ballistic agents                                                                                   |
| rigid              | block agents from moving across                                 | penalizes Lévy agents because they may waste many steps walking against the wall, also not biologically plausible |
| bouncing           | reflect agent trajectories                                      | beneficial for ballistic agents                                                                                   |
| resetting          | make agent choose a new direction when walking against a wall   | probably fairest option                                                                                           |


## Agents
Each agent has:
- a position $(x,y)$
- a direction $\in [0,2\pi]$
- a velocity

At each timestep, an agent's position is updated by converting its direction into $\Delta x$ and $\Delta y$, scaling them by the agent's velocty and the timestep $\Delta t$ and adding the result to the agent's current $x$ and $y$ coordinates.

Perspectives:
- world perspective $[0, 2\pi]$ is used to calculate the direction each agent has in the environment
- ego perspective $[-\pi,\pi]$ is used as a relative orientation measure of the agent itself

Detecting Food Particles:
After each movement, an agent checks for food within its circular body centered at its new position. Since positions are updated in discrete steps, an additional function checks the area between consecutive positions to ensure no food particle is skipped.

There are multiple agent types:
- Ballistic agent (walks straight only)
- Lévy agent (follow Lévy walk like motion patterns)
- Brownian agent (follows brownian motion)
- Reservoir agent (controlled by reservoir computing)

# Parameters
Simulations are configured by the parameters.json file. Here a detailed explanation of the purpose of each parameter:

Agent 
- population_size (int): number of agents
- type (string): identifier of the agent type that is used
- eat_radius (float): size of the agent
- velocity (float): velocity of the agent
- mu (float): $\mu$ of the Lévy agent
- alpha (float): $\alpha$ of the exponential agent
- num_neurons (int): number of neurons in the reservoir
- burn_in_time (int): how long the network activity is simulated without being used after kickstart
- mean (float): mean of the normal distribution that describes the internal weights of the reservoir
- standard_deviation (float): standard deviation of the normal distribution that describes the internal weights of the reservoir

Environment 
- num_food (int): number of food particles in the environment
- size (int): size of the environment
- border_buffer (float): minimum distance between food particles and the border of the environment
- food_buffer (float): minimum distance between food particles
- resetting_boundary (bool): if true, then there are walls around the edges of the environment that the agents cannot cross
- seed (int): seed after which the food particles in the environment are created

Simulation
- total_time (int): total time of the simulation (cannot be changed after Params object has been created)
- delta_t (int): time step increment (needed to translate from velocity units to position units)
- iterations (int): number of iterations for which the simulation is repeated
- save (bool): whether to store the simulation data

# Limitations
- agents can eat at max 1 food particle at each time step
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