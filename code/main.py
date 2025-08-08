from simulation import Simulation
from parameters import Params
from visualization import visualize

def main():
    """
    Run a simulation configured by the parameters in parameters.json.
    """
    # determine where to save the simulation results
    folder = input('Enter simulation name: ')
    params = Params.from_json('parameters.json')

    # execute simulation
    sim = Simulation(params, params.agent)
    _ = sim.run(folder)
    
    # animate results
    visualize(folder)

if __name__ == '__main__':
    main()