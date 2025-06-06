from matplotlib import pyplot as plt
import numpy as np
from met_brewer import met_brew
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import config
import multiprocessing as mp
from agent import ReservoirAgent
from data_io import load_epoch_data, load_population, extract_gif_frames

def visualize(folder):
    """
    Animate all recorded epochs of a simulation.
    Args:
        folder (str): folder where the simulation results are stored
    """
    _, environment, params = load_epoch_data(folder)
    for i in range(0, params.num_epochs // params.intervall_save + 1):
        epoch = 1 if (i == 0) else i * params.intervall_save
        data, _, _ = load_epoch_data(folder, epoch)
        animate(environment, params, data, folder_name=folder, file_name=f'animation_{epoch}')
        if params.agent == ReservoirAgent:
            create_reservoir_activity_plots(folder)

def plot_fitness_log(population_fitness_log, folder, params):
    """
    Plot the average fitness of the population over the epochs.
    """
    if params.num_epochs == 1:
        plt.imshow(population_fitness_log, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel('Agent Index')
        plt.ylabel('Iteration')
        plt.title('Fitness of the Population')
        plt.savefig(config.DATA_PATH / folder / 'fitness_log.png')
        return
    plt.plot(np.sum(population_fitness_log, axis=(0,1)))
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative Fitness')
    plt.title('Fitness Log of the Population')
    folder_path = config.DATA_PATH / folder / 'fitness_log.png'
    plt.savefig(folder_path)

def create_reservoir_activity_plots(folder):
    population = load_population(folder)
    for i, agent in enumerate(population):
        agent.model.plot_activity(folder, i)
        agent.model.plot_eigenvalues_of_weight_matrix(folder, i)

def update(frame, ax, env, params, data, color_dict):
    ax.cla()
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks([0,env.size])
    ax.set_yticks([0,env.size])
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_title(f"Foraging Simulation Environment || t={frame}/{len(data.coords['timestep'])}")
    render_state(ax, data, env, color_dict, params, frame)
    plot_traces(ax, env, params, data, frame, color_dict)

def animate(environment, params, data, folder_name=None, file_name=None):
    with mp.Manager() as manager:
        tqdm_positions = manager.list(range(params.iterations_per_epoch))
        with mp.Pool(config.MAX_PROCESSES) as pool:
            pool.starmap(
                animate_single_iteration,
                [(i, environment, params, data, folder_name, file_name, tqdm_positions[i]) for i in range(params.iterations_per_epoch)]
            )

def animate_single_iteration(i, environment, params, data, folder_name, file_name, tqdm_position, save=True):
    iteration_data = data.sel(iteration=i)

    color_dict = getColorDict()
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_xlim(0, environment.size)
    ax.set_ylim(0, environment.size)
    ax.set_xticks([0, environment.size])
    ax.set_yticks([0, environment.size])
    ax.set_xlabel('X', fontsize = 20)
    ax.set_ylabel('Y', fontsize = 20)
    ax.set_title(f"Foraging Simulation Environment || t=0/{len(data.coords['timestep'])}")

    render_state(ax, iteration_data, environment, color_dict, params, 0)

    frames = tqdm(range(len(iteration_data.coords['timestep'])), desc=f"Animating {i+1}/{params.iterations_per_epoch}", unit="frame", position=tqdm_position, leave=True)
    ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(ax, environment, params, iteration_data, color_dict), interval=100)
    
    if save:
        project_root = Path(__file__).parent.parent
        data_path = project_root / 'data' / folder_name 
        data_path.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            ani.save(filename=data_path / f'animation{i+1}.gif', writer="pillow")
        else: 
            ani.save(filename=data_path / f'{file_name}_it{i+1}.gif', writer="pillow")
        print(f"Safed animation under:\n{data_path}")

def render_state(ax, data, env, color_dict, params, frame):
    plotWall(env, ax, color_dict)
    plotFood(env, ax, color_dict)
    # plot agent perception patches
    percept_matrix = np.vstack((data.sel(timestep=frame)['x_position'].values, data.sel(timestep=frame)['y_position'].values, np.repeat(params.eat_radius, len(data.coords['agent']))))
    plotFilledPatches(env, percept_matrix.transpose(), alpha=0.2, color=color_dict["agent_color"], ax=ax)
    # plot agent eat patches
    eat_matrix = np.vstack((data.sel(timestep=frame)['x_position'].values, data.sel(timestep=frame)['y_position'].values, np.repeat(params.eat_radius, len(data.coords['agent'])), data['ate'].values[frame]))
    plotFilledPatches(env, eat_matrix.transpose(), alpha=0.5, color=color_dict["agent_color"], ax=ax)
    # plot agent directions (the noses that show the direction the agents are heading)
    if frame != 0:
        direction_matrix = np.vstack((data.sel(timestep=frame)['x_position'].values, data.sel(timestep=frame)['y_position'].values, data.sel(timestep=frame)['direction'].values, np.repeat(params.eat_radius*2,len(data.coords['agent']))))
        plot_lines(env, direction_matrix.transpose(), alpha=1, color=color_dict["agent_color"], linewidth=1, ax=ax)

def plotWall(env, ax, color_dict):
    for wall in env.walls:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color=color_dict["food_color"], linewidth=2)

def plotFood(env, ax, color_dict, particle_scale=1):
    if len(env.food_positions) > 0:
        ax.scatter(env.food_positions[:,0], env.food_positions[:,1], color=color_dict["food_color"], label='Food', s=2**particle_scale)

def plotFilledPatches(env, data_matrix, alpha, color, ax):
    # data_matrix = (N,3) with data_matrix[i] = [x_i,y_i,radius_i]
    original_color = color
    # create 50 points along circles circumference
    theta = np.linspace(0, 2*np.pi, 50)

    for i,row in enumerate(data_matrix):
        x = row[0]
        y = row[1]
        r = row[2]

        # plot the number of the agent inside the circle
        ax.text(x, y, str(i), fontsize=8, ha='center', va='center')

        # if there are even 4 rows
        if len(row) == 4 and row[3] != 0:
            color = 'red'
        else:
            color = original_color

        # Normal part
        xn = x + r * np.cos(theta)
        yn = y + r * np.sin(theta)
        ax.fill(xn, yn, color=color, alpha=alpha)

        ### Parts crossing the boundary
        # Overlap with left border?
        if x-r<0: 
            xs = x+env.size + r * np.cos(theta)
            ax.fill(xs, yn, color=color, alpha=alpha)
            # In lower corner?
            if y-r<0:
                xs = x+env.size + r * np.cos(theta)
                ys = y+env.size + r * np.sin(theta)
                plt.fill(xs, ys, color=color, alpha=alpha)
            # In upper corner?
            if y+r>env.size:  
                xs = x+env.size + r * np.cos(theta)
                ys = y-env.size + r * np.sin(theta)
                ax.fill(xs, ys, color=color, alpha=alpha)
        # Overlap with right border?
        if x+r>env.size: 
            xs = x-env.size + r * np.cos(theta)
            ax.fill(xs, yn, color=color, alpha=alpha)
            # In lower corner?
            if y-r<0:
                xs = x-env.size + r * np.cos(theta)
                ys = y+env.size + r * np.sin(theta)
                ax.fill(xs, ys, color=color, alpha=alpha)
            # In upper corner?
            if y+r>env.size:  
                xs = x-env.size + r * np.cos(theta)
                ys = y-env.size + r * np.sin(theta)
                ax.fill(xs, ys, color=color, alpha=alpha)
        # Overlap with lower border?
        if y-r<0: 
            ys = y+env.size + r * np.sin(theta)
            ax.fill(xn, ys, color=color, alpha=alpha)
        # Overlap with upper border?
        if y+r>env.size: 
            ys = y-env.size + r * np.sin(theta)
            ax.fill(xn, ys, color=color, alpha=alpha)

def plot_traces(ax, env, params, data, frame, color_dict):
    # plot agent traces
    number_of_traces = 30 #params.simulation_steps
    i = 1
    velocity = params.velocity
    dt = params.delta_t
    while(frame - i >= 0 and i <= number_of_traces):
        distances = np.repeat(velocity * dt, len(data.coords['agent']))
        trace_matrix = np.column_stack((data.sel(timestep=frame-i)['x_position'].values, data.sel(timestep=frame-i)['y_position'].values, data.sel(timestep=frame-i+1)['direction'].values, distances))
        plot_lines(env, trace_matrix, alpha=1-(i-1)/number_of_traces, color=color_dict["agent_color"], linewidth=0.5, ax=ax)
        i += 1
   
def plot_lines(env, data_matrix, alpha, color, linewidth, ax):
    # data_matrix = (N,4) with data_matrix[i] = [x_i,y_i,dir_i,len_i]
    
    # Check if for each line a color is defined
    multi_colors = False
    if not isinstance(color, (str, np.str_)):
        # If not enough colors given, take first color
        if len(color) != len(data_matrix):
            color = color[0]
        else:
            multi_colors = True
            colors = color

    for i, row in enumerate(data_matrix):
        x = row[0]
        y = row[1]
        direction = row[2]
        length = row[3]

        if multi_colors:
            color = colors[i]

        # Normal part
        xn = np.linspace(x,x+np.cos(direction)*length,10)
        yn = np.linspace(y,y+np.sin(direction)*length,10)
        ax.plot(xn,yn,color=color, alpha=alpha, linewidth=linewidth)

        ### Parts crossing the boundary
        # Overlap with left border?
        if x+np.cos(direction)*length<0: 
            xs = np.linspace(x+env.size, x+env.size+np.cos(direction)*length,10)
            ax.plot(xs,yn,color=color, alpha=alpha, linewidth=linewidth)
            # In lower corner?
            if y+np.sin(direction)*length<0:
                ys = np.linspace(y+env.size, y+env.size+np.sin(direction)*length,10)
                ax.plot(xs,ys,color=color, alpha=alpha, linewidth=linewidth)
            # In upper corner?
            if y+np.sin(direction)*length>env.size:  
                ys = np.linspace(y-env.size, y-env.size+np.sin(direction)*length,10)
                ax.plot(xs,ys,color=color, alpha=alpha, linewidth=linewidth)
        # Overlap with right border?
        if x+np.cos(direction)*length>env.size:
            xs = np.linspace(x-env.size, x-env.size+np.cos(direction)*length,10)
            ax.plot(xs,yn,color=color, alpha=alpha, linewidth=linewidth)
            # In lower corner?
            if y+np.sin(direction)*length<0:
                ys = np.linspace(y+env.size, y+env.size+np.sin(direction)*length,10)
                ax.plot(xs,ys,color=color, alpha=alpha, linewidth=linewidth)
            # In upper corner?
            if y+np.sin(direction)*length>env.size:  
                ys = np.linspace(y-env.size, y-env.size+np.sin(direction)*length,10)
                ax.plot(xs,ys,color=color, alpha=alpha, linewidth=linewidth)
        # Overlap with lower border?
        if y+np.sin(direction)*length<0: 
            ys = np.linspace(y+env.size, y+env.size+np.sin(direction)*length,10)
            ax.plot(xn,ys,color=color, alpha=alpha, linewidth=linewidth)
        # Overlap with upper border?
        if y+np.sin(direction)*length>env.size: 
            ys = np.linspace(y-env.size, y-env.size+np.sin(direction)*length,10)
            ax.plot(xn,ys,color=color, alpha=alpha, linewidth=linewidth)

def getColorDict():

    color_palette1 = np.array(met_brew(name="Tsimshian", n=7, brew_type="continuous"))
    color_palette2 = met_brew(name="VanGogh3", n=8, brew_type='continuous')
    color_palette3 = np.array(met_brew(name="Peru1", n=6, brew_type="continuous"))
    color_palette4 = np.array(met_brew(name="Lakota", n=6, brew_type="continuous"))
    color_palette5 = np.array(met_brew(name="Cassatt1", n=8, brew_type="continuous"))
    color_palette6 = np.array(met_brew(name="OKeeffe2", n=7, brew_type="continuous"))

    agent_color = color_palette4[3]
    agent_colors = list(np.concatenate(([agent_color],color_palette1[[4,5,6]])))
    food_color = color_palette2[-2]
    patch_color = color_palette2[1]
    trained_colors = list([color_palette1[0],color_palette1[1]])
    velo_colors = met_brew(name="Juarez", n=2)
    elitePop_colors = list([color_palette5[0],color_palette5[-1]])
    mean_std_best_colors = list([color_palette6[6],color_palette6[1],color_palette6[4]])
    data_fit1_fit2_colors = list([color_palette3[-1],color_palette3[2],color_palette3[0]])

    # Default case
    color_dict = {"food_color": food_color, # Color of food particles (USED)
                "patch_color": patch_color, # Color of food patches
                "agent_colors": agent_colors, # Colors of [trained, levy1, levy2, brownian] agents
                "agent_color": agent_color,#agent_colors[0], # Color of trained agent
                "trained_colors": trained_colors, # Colors for trained agent comparison
                "velocity_colors": velo_colors,
                "elitePop_colors": elitePop_colors, # Colors for elite vs population comparisons
                "mean_std_best_colors": mean_std_best_colors, # Colors to plot mean + std area around + best individual results
                "data_fit1_fit2_colors": data_fit1_fit2_colors # Colors to plot data + powerlaw fit + exponential fit
                }
    return color_dict

# function to show environment and agents at a given time step
def visualize_state(environment, agents):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_xlim(0, environment.size)
    ax.set_ylim(0, environment.size)
    ax.set_xticks([0,environment.size])
    ax.set_yticks([0,environment.size])
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_title(f'Foraging Simulation Environment', fontsize=15)
    color_dict = getColorDict()
    particle_scale = int(400/environment.size)
    plotFood(environment, ax, color_dict, particle_scale)
    if agents is not None:
        N = len(agents)
        percept_matrix = np.zeros((N,3))
        eat_matrix = np.zeros((N,3))
        direction_matrix = np.zeros((N,4))
        for agent_idx,agent in enumerate(agents):
            percept_matrix[agent_idx] = [agent.position[0],agent.position[1],agent.perception_radius]
            eat_matrix[agent_idx] = [agent.position[0],agent.position[1],agent.eat_radius]
            direction_matrix[agent_idx] = [agent.position[0],agent.position[1],agent.direction,agent.eat_radius*2]
    
        # Plot agent perception patches
        plotFilledPatches(environment,percept_matrix, alpha=0.2, color=color_dict["agent_color"], ax=ax)
        # Plot agent eat patches
        plotFilledPatches(environment,eat_matrix, alpha=0.5, color=color_dict["agent_color"], ax=ax)
        # Plot agent directions
        plot_lines(environment,direction_matrix, alpha=1, color=color_dict["agent_color"], linewidth=1, ax=ax)
    # plt.show()
    return ax