from simulation import Simulation
from parameters import Params
from visualization import visualize, plot_fitness_log

def main():
    """
    Run a simulation configured by the parameters in parameters.json.
    """
    # determine where to save the simulation results
    folder = input('Enter folder name to save simulation results: ')
    params = Params.from_json('parameters.json')

    # execute simulation
    sim = Simulation(params, params.agent)
    fitnesses = sim.run(folder)
    
    # visualize results
    visualize(folder)
    plot_fitness_log(fitnesses, folder, params)

if __name__ == '__main__':
    main()