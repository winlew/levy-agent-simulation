from simulation import Simulation, Params
from visualization import animate, plot_fitness_log
from data_io import load_epoch_data
import datetime

def main():
    """
    Runs a simulation configured by the parameters in parameters.json.
    Saves 
    - environment,
    - parameters and 
    - agent movement logs
    in a folder named after the current date and time.
    Creates an animation of the simulation and a plot of the average population fitness.
    """
    folder = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    params = Params.from_json('parameters.json')
    sim = Simulation(params)
    sim.run(folder)
    data, environment, params = load_epoch_data(folder)
    animate(environment, params, data, folder_name=folder)
    plot_fitness_log(data['average_population_fitness'].values, folder)

if __name__ == '__main__':
    main()