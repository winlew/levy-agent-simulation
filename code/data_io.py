import glob
from agent import ReservoirAgent, LévyAgent
from parameters import Params
import pickle
import numpy as np
import xarray as xr
from pathlib import Path
import config
from PIL import Image
import os
import shutil
import imageio
import json

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
        if (isinstance(agent, ReservoirAgent)):
            with open(folder_path / f'agent_{i}.pkl', 'wb') as f:
                pickle.dump(agent, f)
        if (isinstance(agent, LévyAgent)):
            with open(folder_path / f'agent_{i}.pkl', 'wb') as f:
                pickle.dump(agent, f)

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
    if params.agent == ReservoirAgent or params.agent == LévyAgent:
        files = sorted(glob.glob(str(path) + f'/log/agents/agent_*.pkl'), 
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file in files:
            with open(file, 'rb') as f:
                agent = pickle.load(f)
                population.append(agent)
    else:
        raise NotImplementedError(f"Loading agents of type {params.agent} is not implemented.")
    return population

def initialize_data(params):
    """
    Initializes a xarray dataset to store the simulation results.

    x_position: of each agent at each time step in each iteration
    y_position: of each agent at each time step in each iteration
    direction: of each agent at each time step in each iteration
    ate: 1 if the agent ate at the time step, 0 otherwise 

    Returns:
        xr.Dataset: dataset to store the simulation results
    """
    data = xr.Dataset(
        {
        "x_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations, params.simulation_steps, params.population_size))),
        "y_position": (["iteration", "timestep", "agent"], np.zeros((params.iterations, params.simulation_steps, params.population_size))),
        "direction": (["iteration", "timestep", "agent"], np.zeros((params.iterations, params.simulation_steps, params.population_size))),
        "ate": (["iteration", "timestep", "agent"], np.zeros((params.iterations, params.simulation_steps, params.population_size))),
        },
        coords={
            "iteration": np.arange(params.iterations),
            "timestep": np.arange(params.simulation_steps),
            "agent": np.arange(params.population_size),
        }
    )
    return data
    
def update_data(data, iteration, trajectory_log):
    """
    Updates the data with the results of a single iteration.
    
    Args:
        data (xr.Dataset): dataset to store the simulation results
        iteration (int): index of the iteration
        trajectory_log (np.array): trajectory of the agents in the current iteration
    """
    data["x_position"].loc[iteration, :, :] = trajectory_log[:, :, 0]
    data["y_position"].loc[iteration, :, :] = trajectory_log[:, :, 1]
    data["direction"].loc[iteration, :, :] = trajectory_log[:, :, 2]
    data["ate"].loc[iteration, :, :] = trajectory_log[:, :, 3]

def extract_agents(folder, agent_indexes):
    """
    Trim the data to only contain the selected agents and safe it in a new folder.
    """
    safe_folder = f"{folder}_{len(agent_indexes)}_extracted"
    data, environment, _ = load_data(folder)
    trimmed_data = data.isel(agent=agent_indexes)
    save_simulation_context(safe_folder, environment)
    save_data(safe_folder, trimmed_data, None)
    params_file = config.DATA_PATH / safe_folder / 'parameters.json'
    with open(params_file, 'r') as f:
        params_dict = json.load(f)
    params_dict['agent']['population_size'] = len(agent_indexes)
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    return safe_folder
    
def save_simulation_context(folder, environment):
    """
    Saves the immutable settings:
    - the environment
    - simulation parameters.

    Args:
        folder (str): folder to save the simulation context in
        environment (Environment): environment of the simulation
    """
    folder_path = config.DATA_PATH / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / 'environment.pkl', 'wb') as f:
        pickle.dump(environment, f)
    shutil.copyfile(config.PROJECT_ROOT_PATH / 'code/parameters.json', folder_path / 'parameters.json')

def save_data(folder, data, population):
    """
    Make a snapshot of the simulation results:
    - agent motion
    - population.
    
    Args:
        folder (str): folder to save the simulation results in
        data (xr.Dataset): dataset with the simulation results of all iterations
        population (list): list of agents
    """
    folder_path = config.DATA_PATH / folder / 'log'
    folder_path.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(folder_path / f'motion.nc')
    if population:
        save_population(population, folder + '/log/agents')

def load_data(folder):
    """
    Loads:
    - environment
    - agent motion data
    - parameters
    for a given simulation.
    """
    params = load_parameters(folder)
    folder_path = config.DATA_PATH / folder
    data = xr.open_dataset(folder_path / 'log/motion.nc')
    with open(folder_path / 'environment.pkl', 'rb') as f:
        environment = pickle.load(f)
    return data, environment, params

def load_parameters(folder):
    """
    Load the simulation parameters.
    """
    folder_path = config.DATA_PATH / folder
    params = Params.from_json(folder_path / 'parameters.json')  
    return params

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

def combine_gifs_side_by_side(path1, path2, output_path):
    """
    Place gifs side by side

    Args:
        path1 (str): path to the first gif
        path2 (str): path to the second gif
        output_path (str): path to save the combined gif at
    """
    gif1 = imageio.get_reader(path1)
    gif2 = imageio.get_reader(path2)
    number_of_frames = min(gif1.get_length(), gif2.get_length())
    new_gif = imageio.get_writer('output.gif')
    for frame_number in range(number_of_frames):
        img1 = gif1.get_data(frame_number)
        img2 = gif2.get_data(frame_number)
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)
    gif1.close()
    gif2.close()
    new_gif.close()
    shutil.move('output.gif', output_path)
