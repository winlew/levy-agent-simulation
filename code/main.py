from simulation import Simulation
from parameters import Params
from visualization import visualize, plot_fitness_log
from data_io import extract_gif_frames

def main():
    """
    Run a simulation configured by the parameters in parameters.json.
    """
    # determine where to save the simulation results
    folder = input('Enter simulation name: ')
    params = Params.from_json('parameters.json')

    # execute simulation
    sim = Simulation(params, params.agent)
    fitnesses = sim.run(folder)
    
    # visualize results
    visualize(folder)
    plot_fitness_log(fitnesses, folder)
    extract_gif_frames(folder, 'animation_1.gif')

if __name__ == '__main__':
    main()