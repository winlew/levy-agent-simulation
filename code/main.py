from simulation import Simulation, Params
from visualization import animate 
from data_io import load_epoch_data, extract_gif_frames
from agent import *

def main():
    """
    Runs a simulation configured by the parameters in parameters.json.
    Saves 
    - environment,
    - parameters and 
    - agent movement data
    in a folder named after the current date and time.
    Creates an animation of the simulation and a plot of the average population fitness.
    """
    folder = input('Enter folder name to save simulation results under: ')
    params = Params.from_json('parameters.json')

    # choose agent type here
    agent = LévyAgent
    # for more complex agents they need model too, but can be part of their class
    # for the agents that don't learn we can do a single epoch with many iterations
    # for the agents that do learn we can to multiple epochs
    
    # execute simulation
    sim = Simulation(params, agent)
    sim.run(folder)
    # visualize results
    data, environment, params = load_epoch_data(folder)
    animate(environment, params, data, folder_name=folder)
    extract_gif_frames(folder, 'animation1.gif')

if __name__ == '__main__':
    main()