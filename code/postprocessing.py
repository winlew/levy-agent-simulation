from visualization import extract_high_resolution_frame, visualize
from data_io import extract_agents, extract_gif_frames

if __name__ == '__main__':
    folder = input('Enter name of folder to post process: ')
    mode = input("Type:\n 'f' to extract frame " \
                            "\n 'a' to extract agent trajectories"
                            "\n 'q' to quit\n")

    while (mode != 'q'):
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
            extract_gif_frames(extracted_folder, 'animation_1.gif')
            if not extracted_folder:
                print('failed')
                exit()
            visualize(extracted_folder)
            print('success')
        else:
            print('No action connected to input.')
        mode = input("Type:\n 'f' to extract frame" \
                          "\n 'a' to extract agent trajectories"
                          "\n 'q' to quit.\n")