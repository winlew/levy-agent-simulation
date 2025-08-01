from visualization import extract_high_resolution_frame, visualize
from data_io import extract_agents

if __name__ == '__main__':
    folder = input('Enter name of folder to post process: ')

    mode = input("Type 'f' to extract frame or type 'a' to extract agent trajectories: ")

    if mode == 'f':
        frame = int(input('Timestep: '))
        iteration = int(input('From iteration: '))
        extract_high_resolution_frame(folder, frame, iteration)
        print('success')
    elif mode == 'a':
        agent_indexes = []
        while len(agent_indexes) < 10:
            agent_indexes.append(int(input(f'Enter agent index ({len(agent_indexes)}/10): ')))
        extracted_folder = extract_agents(folder, agent_indexes)
        if not extracted_folder:
            print('failed')
            exit()
        visualize(extracted_folder)
        print('success')
    else:
        print('No action connected to input.')