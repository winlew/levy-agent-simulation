import torch
import glob
from agent import RnnAgent, Rnn, Agent
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

    Args:
        population (list): list of agents
        folder (str): folder to save the population in
    """
    folder_path = config.DATA_PATH / folder 
    folder_path.mkdir(parents=True, exist_ok=True)
    for i, agent in enumerate(population):
        if agent.model:
            torch.save(agent.model.state_dict(), folder_path / f'agent_{i}.pth')

def load_population(folder):
    """
    Loads a population from files and returns it.

    Args:
        folder (str): folder to load the population from

    Returns:
        population (list): list of agents
    """
    params = load_parameters(folder)
    population = []
    path = Path(config.DATA_PATH) / folder    
    for file in glob.glob(str(path / f'population{params.num_epochs}/agent_*.pth')):
        model = Rnn(params)
        model.load_state_dict(torch.load(file, weights_only=False))
        agent = RnnAgent(params, model=model)
        population.append(agent)
    return population

def initialize_epoch_data(params):
    """
    Initializes a xarray dataset to store the simulation results of a single epoch.

    x_position: of each agent at each time step in each iteration
    y_position: of each agent at each time step in each iteration
    direction: of each agent at each time step in each iteration
    ate: 1 if the agent ate at the time step, 0 otherwise 

    Returns:
        xr.Dataset: dataset to store the simulation results
    """
    data = xr.Dataset(
        {
        "x_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "y_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "direction": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        "ate": (["iteration", "timestep", "agent"], np.zeros((params.iterations_per_epoch, params.simulation_steps, params.population_size))),
        },
        coords={
            "iteration": np.arange(params.iterations_per_epoch),
            "timestep": np.arange(params.simulation_steps),
            "agent": np.arange(params.population_size),
        }
    )
    return data
    
def update_epoch_data(data, iteration, trajectory_log):
    """
    Updates the data for a single epoch with the results of a single iteration.
    
    Args:
        data (xr.Dataset): dataset to store the simulation results
        iteration (int): index of the iteration
        trajectory_log (np.array): trajectory of the agents in the current iteration
    """
    data["x_position"].loc[iteration, :, :] = trajectory_log[:, :, 0]
    data["y_position"].loc[iteration, :, :] = trajectory_log[:, :, 1]
    data["direction"].loc[iteration, :, :] = trajectory_log[:, :, 2]
    data["ate"].loc[iteration, :, :] = trajectory_log[:, :, 3]

def save_simulation_context(folder, environment, params):
    """
    Saves the immutable settings:
    - the environment
    - simulation parameters.

    Args:
        folder (str): folder to save the simulation context in
        environment (Environment): environment of the simulation
        params (Params): parameters of the simulation
    """
    folder_path = config.DATA_PATH / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / 'environment.pkl', 'wb') as f:
        pickle.dump(environment, f)
    with open(folder_path / 'parameters.pkl', 'wb') as f:
        pickle.dump(params, f)
    write_parameters_to_text(params, folder)

def save_epoch_data(folder, data, population, epoch):
    """
    Make a snapshot of the results of a single epoch:
    - agent motion
    - population.
    
    Args:
        folder (str): folder to save the simulation results in
        data (xr.Dataset): dataset with the simulation results of all iterations
        population (list): list of agents
        epoch (int): index of the epoch
    """
    folder_path = config.DATA_PATH / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(folder_path / f'epoch_{epoch}.nc')
    if population:
        save_population(population, folder + f'/population{epoch}')

def load_epoch_data(folder, epoch=None):
    """
    Loads:
    - environment
    - agent motion data
    - parameters
    from a single epoch of a simulation.
    If epoch is not None, load the data of a specific epoch.
    """
    params = load_parameters(folder)
    folder_path = config.DATA_PATH / folder
    if epoch is None:
        data = xr.open_dataset(folder_path / f'epoch_{params.num_epochs}.nc')
    else:
        data = xr.open_dataset(folder_path / f'epoch_{epoch}.nc')
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
    params_dict = {key: value for key, value in params.__dict__.items() if key != 'agent'} 
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
    extract_gif_frames('exploration_study', 'levy.gif')
    pass