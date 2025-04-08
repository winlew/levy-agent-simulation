from simulation import Simulation
from config import Params
from visualization import visualize, plot_fitness_log

def main():
    """
    Runs a simulation configured by the parameters in parameters.json.
    Saves 
    - environment,
    - parameters and 
    - agent motion.
    Creates an animation of the last epoch of the simulation and plots the average population fitness.
    """
    folder = input('Enter folder name to save simulation results under: ')
    params = Params.from_json('parameters.json')

    # execute simulation
    sim = Simulation(params, params.agent)
    mean_fitness_per_epoch = sim.run(folder)
    # visualize results
    visualize(folder)
    plot_fitness_log(mean_fitness_per_epoch, folder)

if __name__ == '__main__':
    main()