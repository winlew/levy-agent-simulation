import torch
import glob
from agent import Agent, Rnn
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
import config
import json
from PIL import Image
import os

def save_population(population, folder):
    """
    Saves a population to files.
    """
    folder_path = config.DATA_PATH / folder 
    folder_path.mkdir(parents=True, exist_ok=True)
    for i, agent in enumerate(population):
        torch.save(agent.model.state_dict(), folder_path / f'agent_{i}.pth')

def load_population(folder):
    """
    Loads a population from files and returns it.
    """
    params = load_parameters(folder)
    population = []
    path = Path(config.DATA_PATH) / folder    
    for file in glob.glob(str(path / f'population{params.num_epochs}/agent_*.pth')):
        model = Rnn(params)
        model.load_state_dict(torch.load(file, weights_only=False))
        agent = Agent(model, params)
        population.append(agent)
    return population

def initialize_epoch_data(params):
    """
    Initializes a xarray dataset to store the simulation results or a single epoch.

    x_position: of each agent at each time step in each iteration
    y_position: of each agent at each time step in each iteration
    direction: of each agent at each time step in each iteration
    perception_radius: of each agent at each time step in each iteration
    meals: number of food items eaten by each agent in each iteration
    meal_timeline: 1 for each time step in which the agent ate, 0 otherwise (for every agent)
    average_population_fitness: average fitness of the population in each epoch

    Returns:
        xr.Dataset: dataset to store the simulation results
    """
    data = xr.Dataset(
        {
        "x_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "y_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "direction": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "perception_radius": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "meals": (["iteration", "agent"], np.zeros((params.iterations_per_epoch, params.population_size))),
        "meal_timeline": (["iteration", "ate", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "average_population_fitness": (["epoch"], np.zeros(params.num_epochs)),
        },
        coords={
            "iteration": np.arange(params.iterations_per_epoch),
            "timestep": np.arange(params.simulation_steps),
            "agent": np.arange(params.population_size),
            "epoch": np.arange(params.num_epochs),
        }
    )
    return data
    
def update_epoch_data(data, iteration, trajectory_log, meals_per_iteration):
    """
    Meant to be called to update the data for a single epoch in each iteration.
    """
    data["x_position"].loc[iteration, :, :] = trajectory_log[:, :, 0]
    data["y_position"].loc[iteration, :, :] = trajectory_log[:, :, 1]
    data["direction"].loc[iteration, :, :] = trajectory_log[:, :, 2]
    data["perception_radius"].loc[iteration, :, :] = trajectory_log[:, :, 3]
    data["meals"].loc[:, :] = meals_per_iteration

def update_meal_timelines(data, iteration, population):
    meal_timeline = np.zeros((len(population[0].meal_timeline), len(population)))
    for i, agent in enumerate(population):
        meal_timeline[:, i] = agent.meal_timeline
    data['meal_timeline'].loc[iteration, :, :] = meal_timeline

def update_fitness_log(data, population_fitness_log):
    """
    Log average fitness of the population in each epoch.
    """
    data["average_population_fitness"].loc[:] = population_fitness_log

def save_simulation_context(folder, environment, params):
    """
    Saves the immutable settings of the simulation.
    - the environment
    - simulation parameters
    """
    folder_path = config.DATA_PATH / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / 'environment.pkl', 'wb') as f:
        pickle.dump(environment, f)
    with open(folder_path / 'parameters.pkl', 'wb') as f:
        pickle.dump(params, f)

def save_epoch_data(folder, data, population, epoch):
    """
    Save the results of a single epoch.
    - agent movement and food consumption
    - population
    """
    folder_path = config.DATA_PATH / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(folder_path / f'agent_data{epoch}.nc')
    if population:
        save_population(population, folder + f'/population{epoch}')

def load_epoch_data(folder, epoch=None):
    """
    Load the simulation results.
    Loads:
        - the environment
        - agent data
        - parameters
    If epoch is not None, load the data of a specific epoch.
    """
    params = load_parameters(folder)
    folder_path = config.DATA_PATH / folder
    if epoch is None:
        data = xr.open_dataset(folder_path / f'agent_data{params.num_epochs}.nc')
    else:
        data = xr.open_dataset(folder_path / f'agent_data{epoch}.nc')
    with open(folder_path / 'environment.pkl', 'rb') as f:
        environment = pickle.load(f)
    return data, environment, params

def load_parameters(folder):
    """
    Load the simulation parameters.
    """
    folder_path = config.DATA_PATH / folder
    with open(folder_path / 'parameters.pkl', 'rb') as f:
        params = pickle.load(f)
    return params

def write_parameters_to_text(params, folder):
    """
    Writes the parameters from a JSON file to a text file for quick lookup.
    """
    folder_path = config.DATA_PATH / folder / 'parameters.txt'
    params_dict = params.__dict__
    with open(folder_path, 'w') as text_file:
        json.dump(params_dict, text_file, indent=4)

def extract_gif_frames(folder, file_name):
    """
    Extracts the frames from a gif file.
    """
    path = config.DATA_PATH / folder / file_name
    gif = Image.open(path)

    output_folder = config.DATA_PATH / folder / f"frames_{file_name}"
    output_folder.mkdir(parents=True, exist_ok=True)

    for i in range(gif.n_frames):
        gif.seek(i)
        gif.save(os.path.join(output_folder, f"frame_{i}.png"))

if __name__ == '__main__':
    extract_gif_frames('exploration_study', 'ballistic.gif')
    pass