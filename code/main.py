from simulation import Simulation
from config import Params
from visualization import visualize, plot_fitness_log

def main():
    """
    Runs a simulation configured by the parameters in parameters.json.
    """
    # determine where to save the simulation results
    folder = input('Enter folder name to save simulation results under: ')
    params = Params.from_json('parameters.json')

    # execute simulation
    sim = Simulation(params, params.agent)
    mean_fitness_per_epoch = sim.run(folder)
    
    # visualize results
    visualize(folder)
    plot_fitness_log(mean_fitness_per_epoch, folder, params)

if __name__ == '__main__':
    main()