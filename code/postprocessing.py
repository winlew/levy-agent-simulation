from visualization import extract_high_resolution_frame, visualize
from data_io import extract_agents, extract_gif_frames, combine_agents

if __name__ == '__main__':
    # There are three postprocessing procedures. 
    #   1. Extract High Resolution Frame
    #   2. Extract Agents
    #   3. Combine Agent Trajectories from Multiple Runs

    folder = 'COMB'
    frame = 265
    iteration = 0
    extract_high_resolution_frame(folder, frame, iteration)
    # ---
    # folder = '003700_EXT'
    # frame = 129
    # iteration = 0
    # extract_high_resolution_frame(folder, frame, iteration)

    # folder = '003700'
    # agent_indexes = [10, 8, 2, 36, 3, 5, 45, 16, 11, 34]
    # extracted_folder = extract_agents(folder, agent_indexes)
    # visualize(extracted_folder)
    # extract_gif_frames(extracted_folder, 'animation_1.gif')

    # folders = ['003160_1', '003160_2', '003160_3', '003160_4']
    # agent_indexes = [[8, 18, 19, 6, 20], [27, 11], [88, 85], [12]]
    # combine_agents(folders, agent_indexes, 'COMB')
    # visualize('COMB')
    # extract_gif_frames('COMB', 'animation_1.gif')