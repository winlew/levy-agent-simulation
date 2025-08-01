from simulation import Simulation
from parameters import Params
from visualization import visualize, plot_fitness_log
from data_io import extract_gif_frames

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
    plot_fitness_log(fitnesses, folder)
    extract_gif_frames(folder, 'animation_1.gif')

if __name__ == '__main__':
    main()

    # from visualization import draw_frame_in_high_resolution
    # draw_frame_in_high_resolution('00', 50, 0)

    # folder = '003130_Popular'
    # from data_io import extract_agents
    # extracted_folder = extract_agents(folder, [9,11,2,40,48,3,21,18,17,41]) #26
    # visualize(extracted_folder)