from matplotlib import pyplot as plt
import numpy as np
from met_brewer import met_brew
import matplotlib.animation as animation
from pathlib import Path
from tqdm import tqdm
import config
import multiprocessing as mp
from agent import ReservoirAgent, LévyAgent
from data_io import load_data, load_population, extract_gif_frames
from config import DATA_PATH
from environment import Environment
from parameters import Params
from matplotlib.animation import FuncAnimation
import igraph as ig

def visualize(folder):
    """
    Animate a simulation.
    Args:
        folder (str): folder where the simulation results are stored
    """
    data, environment, params = load_data(folder)
    animate(environment, params, data, folder_name=folder)
    if params.agent == ReservoirAgent:
        visualize_reservoir(folder)
        plot_step_length_distribution_of_agents(folder)
    if params.agent == LévyAgent:
        plot_step_length_distribution_of_agents(folder)

def plot_fitness_log(fitnesses, folder):
    """
    Visualize population fitness.
    """
    plt.clf()
    plt.imshow(fitnesses, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Agent Index')
    plt.ylabel('Iteration')
    plt.title('Fitness of the Population')
    plt.savefig(config.DATA_PATH / folder / 'fitness_log.png')
    return

def visualize_reservoir(folder):
    population = load_population(folder)
    for i, agent in enumerate(population):
        plot_activity(agent.model, folder, i)
        plot_eigenvalues_of_weight_matrix(agent.model, folder, i)
        plot_reservoir_outputs(agent, folder, i)
        # not worth the effort
        # plot_weights(agent.model, folder, i)
        # draw_reservoir_graph(agent.model, folder, i)


def update(frame, ax, env, params, data, color_dict):
    ax.cla()
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks([0,env.size])
    ax.set_yticks([0,env.size])
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_title(f"t={frame}/{len(data.coords['timestep'])}")
    render_state(ax, data, env, color_dict, params, frame)
    plot_traces(ax, env, params, data, frame, color_dict)

def animate(environment, params, data, folder_name=None):
    with mp.Manager() as manager:
        tqdm_positions = manager.list(range(params.iterations))
        with mp.Pool(config.MAX_PROCESSES) as pool:
            pool.starmap(
                animate_single_iteration,
                [(i, environment, params, data, folder_name, tqdm_positions[i]) for i in range(params.iterations)]
            )

def animate_single_iteration(i, environment, params, data, folder_name, tqdm_position, save=True):
    iteration_data = data.sel(iteration=i)

    color_dict = get_color_dict()
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.set_xlim(0, environment.size)
    ax.set_ylim(0, environment.size)
    ax.set_xticks([0, environment.size])
    ax.set_yticks([0, environment.size])
    ax.set_xlabel('X', fontsize = 20)
    ax.set_ylabel('Y', fontsize = 20)
    ax.set_title(f"t=0/{len(data.coords['timestep'])}")

    render_state(ax, iteration_data, environment, color_dict, params, 0)

    frames = tqdm(range(len(iteration_data.coords['timestep'])), desc=f"Animating {i+1}/{params.iterations}", unit="frame", position=tqdm_position, leave=True)
    ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(ax, environment, params, iteration_data, color_dict), interval=100)
    
    if save:
        project_root = Path(__file__).parent.parent
        data_path = project_root / 'data' / folder_name 
        data_path.mkdir(parents=True, exist_ok=True)
        ani.save(filename=data_path / f'animation_{i+1}.gif', writer="pillow")

def render_state(ax, data, env, color_dict, params, frame):
    plotWall(env, ax, color_dict)
    plotFood(env, ax, color_dict)
    # plot agent eat patches
    eat_matrix = np.vstack((data.sel(timestep=frame)['x_position'].values, data.sel(timestep=frame)['y_position'].values, np.repeat(params.eat_radius, params.population_size), data['ate'].values[frame]))
    plotFilledPatches(env, eat_matrix.transpose(), alpha=0.5, color=color_dict["agent_color"], ax=ax)
    # plot agent directions (the noses that show the direction the agents are heading)
    if frame != 0:
        direction_matrix = np.vstack((data.sel(timestep=frame)['x_position'].values, data.sel(timestep=frame)['y_position'].values, data.sel(timestep=frame)['direction'].values, np.repeat(params.eat_radius*2, params.population_size)))
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

def plot_traces(ax, env, params, data, frame, color_dict, number_of_traces = 60, fade=True):
    """
    Plot the agent traces for the last `number_of_traces` time steps.
    Set 'number_of_traces' to params.simulation_steps to plot all traces.    
    """
    i = 1
    velocity = params.velocity
    dt = params.delta_t
    while(frame - i >= 0 and i <= number_of_traces):
        if fade:
            opacity = 1-(i-1)/number_of_traces
        else:
            opacity = 1
        distances = np.repeat(velocity * dt, params.population_size)
        trace_matrix = np.column_stack((data.sel(timestep=frame-i)['x_position'].values, data.sel(timestep=frame-i)['y_position'].values, data.sel(timestep=frame-i+1)['direction'].values, distances))
        plot_lines(env, trace_matrix, alpha=opacity, color=color_dict["trace_color"], linewidth=0.5, ax=ax)
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

def get_color_dict():
    """
    Returns a dictionary with colors from the met_brew package for the visualization.
    """
    color_palette = met_brew(name='Troy', n=8, brew_type='continuous')
    agent_color = "#C86B52"
    trace_color = "#596f7a"
    food_color = '#000000'
    color_dict = {"food_color": food_color,
                  "agent_color": agent_color,
                  "trace_color": trace_color}
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
    ax.set_title(f'Simulation Environment', fontsize=15)
    color_dict = get_color_dict()
    plotFood(environment, ax, color_dict)
    if agents is not None:
        N = len(agents)
        eat_matrix = np.zeros((N,3))
        direction_matrix = np.zeros((N,4))
        for agent_idx,agent in enumerate(agents):
            eat_matrix[agent_idx] = [agent.position[0],agent.position[1],agent.eat_radius]
            direction_matrix[agent_idx] = [agent.position[0],agent.position[1],agent.direction,agent.eat_radius*2]
    
        # Plot agent eat patches
        plotFilledPatches(environment,eat_matrix, alpha=0.5, color=color_dict["agent_color"], ax=ax)
        # Plot agent directions
        plot_lines(environment,direction_matrix, alpha=1, color=color_dict["agent_color"], linewidth=1, ax=ax)
    # plt.show()
    return ax

def plot_step_length_distribution_of_agents(folder, tolerance=0.001):
    """
    Plot how the step lengths of all agents are distributed.
    - create regular histogram
    - create log-binned histogram
    The step lengths is defined as the number of time steps an agent moves straight in a certain direction.

    Args:
        folder (str): folder where the simulation results are stored
        tolerance (float): threshold to determine whether an agent is moving straight or not
    """
    _, _, params = load_data(folder)
    population = load_population(folder)

    ballistic_movement_detected = False

    step_lengths = np.array([])
    if params.agent == LévyAgent:
        for agent in population:
            step_lengths = np.concatenate((step_lengths, agent.step_length_log))
    elif params.agent == ReservoirAgent:
        for agent in population:
            step_counter = 1
            for output in agent.output_log:
                if abs(output) > 1 - tolerance or abs(output) < tolerance:
                    step_counter += 1
                else:
                    step_lengths = np.append(step_lengths, step_counter)
                    step_counter = 1
            step_lengths = np.append(step_lengths, step_counter)
            if step_lengths.size == 0:
                ballistic_movement_detected = True
    else:
        raise ValueError(f"This function is not implemented for {params.agent}.")

    if ballistic_movement_detected:
        return
    
    counts = np.bincount(step_lengths.astype(int))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(counts)), counts, color='blue', alpha=0.7)
    plt.title(f'Histogram of Step Lengths')
    plt.xlabel('Step Length')
    plt.ylabel('Number of Occurences')
    path = Path(DATA_PATH) / folder 
    path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path / 'step_length_distribution.png')

    plt.clf()
    plt.figure(figsize=(8, 8))
    counts, bins = np.histogram(step_lengths, bins=[2**i for i in range(int(max(step_lengths)).bit_length() + 1)])
    plt.loglog(bins[:-1], counts, marker='o', linestyle='none')
    plt.grid(True, which="both", ls="--")
    plt.title(f'Log-Binned Step Length Distribution')
    plt.xlabel('Step Length')
    plt.ylabel('Number of Occurences')
    plt.tight_layout()
    plt.savefig(path / 'log_binned_step_length_distribution.png')

def extract_agent_trajectory(folder, iteration, agent_number, buffer = 5):
    """
    Isolate the trajectory of a certain agent in an empty environment.
    """
    iteration -= 1
    # load data
    data, _, params = load_data(folder)
    single_agent_data = data.sel(agent = agent_number)
    x_positions = data['x_position'].values[iteration, :, agent_number]
    y_positions = data['y_position'].values[iteration, :, agent_number]
    x_min, x_max = x_positions.min(), x_positions.max()
    y_min, y_max = y_positions.min(), y_positions.max()
    single_agent_data['x_position'].values = single_agent_data['x_position'].values - x_min + buffer
    single_agent_data['y_position'].values = single_agent_data['y_position'].values - y_min + buffer
    # set up parameters
    params = Params(
        num_food = 0,
        size = max(x_max - x_min, y_max - y_min) + 2*buffer,
        velocity = params.velocity,
        eat_radius = params.eat_radius,
        mu = 2,
        alpha = 1,
        iterations = 1,
        population_size = 1,
        total_time = params.total_time,
        delta_t = params.delta_t,
        border_buffer = params.border_buffer,
        food_buffer = params.food_buffer,
        seed = 10,
        resetting_boundary = False
    )
    # TODO agent should not be colored red when eating
    # single_agent_data['ate'] = np.zeros((params.iterations, params.simulation_steps, 1)),
    environment = Environment(params)
    animate_single_iteration(iteration, environment, params, single_agent_data, folder + f'/isolated_agent_{agent_number}', 0, save=True)

def draw_reservoir_graph(reservoir, folder, id, percent_threshold=0.08):
    """
    Visualize the neural reservoir as a graph.
    Show only the most important weights.

    Args:
        reservoir (Reservoir): a reservoir of neurons
        folder (str): where to store the plot
        id (int): number of the agent
        percent_treshold (float): only the top percent_threshold highest weights are drawn
    """
    matrix = reservoir.weight_matrix
    flat = np.abs(matrix.flatten())
    threshold = np.percentile(flat, 100 - percent_threshold)
    adjacency = (np.abs(matrix) >= threshold).astype(int)
    g = ig.Graph.Adjacency(adjacency.tolist(), mode='directed')
    layout = g.layout_fruchterman_reingold()
    _, ax = plt.subplots(figsize=(15, 15))
    ig.plot(
        g,
        target=ax,
        layout=layout,
        vertex_color="#FF6B6B96",
        vertex_size=5,
        vertex_label_size=8,
        edge_width=0.5,
        edge_color="#3B3B3BFF",
        edge_arrow_size=3,
        edge_arrow_width=2,
        margin=0
    )
    path = Path(DATA_PATH) / folder / 'reservoir_graphs'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f'agent_{id}.svg', format='svg')
    plt.close()

def plot_weights(reservoir, folder, id):
    """
    Plot a heatmap of the weight matrix.

    Args
        reservoir (Reservoir): a reservoir of neurons
        folder (str): where to store the plot
        id (int): number of the agent
    """
    _, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.imshow(reservoir.weight_matrix, cmap='coolwarm', aspect='auto', interpolation='none')
    axes.set_ylabel('neuron #')
    axes.set_xlabel('neuron #')
    path = Path(DATA_PATH) / folder / 'weight_matrices'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f'agent_{id}.svg', format='svg')
    plt.close()

def plot_activity(reservoir, folder, id):
    """
    Plot the activity of each neuron over time.

    Args
        reservoir (Reservoir): a reservoir of neurons
        folder (str): where to store the plot
        id (int): number of the agent
    """
    plt.rcParams.update({'font.size': 16})
    _, ax = plt.subplots(figsize=(10, 5))
    activity = np.concatenate((reservoir.burn_in_state_matrix[:-1], reservoir.neuron_state_time_matrix), axis=0)
    im = ax.imshow(activity.T, cmap='seismic', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Activity', labelpad=5, rotation=90, loc='center')
    cbar.ax.yaxis.set_label_position('left')
    ax.set_ylabel('Neuron')
    ax.set_xlabel('Time')
    burn_in_end = reservoir.burn_in_state_matrix.shape[0]
    ax.scatter(burn_in_end, activity.shape[1] - 0.5 + 15, marker='v', s=30, color='black', zorder=10, clip_on=False, label='Burn-in End')
    ax.set_ylim(-0.5, activity.shape[1] - 0.5)
    path = Path(DATA_PATH) / folder / 'reservoir_activities'
    path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path / f'agent_{id}.svg', format='svg')
    plt.close()

# TODO there is a newer function in the study to calculate them
def plot_eigenvalues_of_weight_matrix(reservoir, folder, id):
    """
    Plot the eigenvalues of the weight matrix in the complex plane.

    Args
        reservoir (Reservoir): a reservoir of neurons
        folder (str): where to store the plot
        id (int): number of the agent
    """
    eigenvalues, _ = np.linalg.eig(reservoir.weight_matrix)
    spectral_radius = max(abs(eigenvalues))
    plt.figure(figsize=(8, 8))
    plt.scatter(eigenvalues.real, eigenvalues.imag, s=10)
    plt.xlabel(r'$\mathrm{Re}(\lambda)$')
    plt.ylabel(r'$\mathrm{Im}(\lambda)$')
    plt.text(0.95, 0.95, f'$\\rho = {spectral_radius:.3f}$', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.5))
    plt.axhline(0, color='black', lw=0.2, ls='--')
    plt.axvline(0, color='black', lw=0.2, ls='--')
    circle = plt.Circle((0, 0), spectral_radius, color='grey', ls='--', fill=False, lw=1)
    plt.gca().add_artist(circle)
    plt.grid(linewidth=0.3)
    plt.xlim(-1.5 * spectral_radius, 1.5 * spectral_radius)
    plt.ylim(-1.5 * spectral_radius, 1.5 * spectral_radius)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    path = Path(DATA_PATH) / folder / 'eigenvalues'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f'agent_{id}.svg', format='svg')
    plt.close()

def plot_reservoir_outputs(agent, folder, id):
    """
    Plots the reservoir outputs over time.

    Args
        agent (ReservoirAgent): an agent that is controlled by a neural reservoir
        folder (str): where to store the plot
        id (int): number of the agent
    """
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(15, 5))
    plt.scatter(range(len(agent.output_log)), agent.output_log, s=10)
    plt.xlabel('Time Step')
    plt.ylabel('Reservoir Output')
    path = Path(DATA_PATH) / folder / 'reservoir_outputs'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f'agent_{id}.svg', format='svg')
    plt.tight_layout()
    plt.close()

# TODO has to get a rework
def animate_reservoir(reservoir, file_name):
    """
    Animates the network activity over time in a graph.

    Args:
        reservoir (Reservoir): a neural reservoir
        file_name (str): name of the output GIF file
        folder (str): folder to save the GIF in
    """
    time_steps = reservoir.neuron_state_time_matrix.shape[0]

    g = ig.Graph.Adjacency(np.abs(reservoir.weight_matrix).tolist(), mode='directed')
    # layout = g.layout(layout='auto')
    layout = g.layout_fruchterman_reingold()

    # COLOR WEIGHTS AND MAKE STRONGER WEIGHT CONNECTIONS SHORTER
    # shift to positive, by adding the smallest negative weight, then divide by the range to scale to [0, 1]
    normalized_weights = (reservoir.weight_matrix - reservoir.weight_matrix.min()) / (reservoir.weight_matrix.max() - reservoir.weight_matrix.min())
    # Important: For distance-based layouts, SMALLER values should pull nodes CLOSER
    # So we need to INVERT the weights for edges (higher weight = shorter distance)
    # We'll use a small epsilon to avoid division by zero for zero weights
    epsilon = 0.0001
    edge_distances = []
    for i, edge in enumerate(g.es):
        source, target = edge.source, edge.target
        # Higher normalized weight = shorter distance (closer nodes)
        # Invert and scale: smaller values will pull nodes closer in the layout
        edge_distance = 1.0 - normalized_weights[source][target] + epsilon
        edge_distances.append(edge_distance)
        # Update edge attributes
        edge["distance"] = edge_distance
        edge["width"] = 1 + 5 * normalized_weights[source][target]  # Thicker lines for stronger connections
        # Set edge color based on weight
        if normalized_weights[source][target] > 0.66:
            edge["color"] = "darkred"  # Strongest connections
        elif normalized_weights[source][target] > 0.33:
            edge["color"] = "red"      # Medium-strong connections
        else:
            edge["color"] = "pink"     # Weaker connections

    # COLORIZE NODES BASED ON STATES
    min_state = reservoir.neuron_state_time_matrix.min()
    max_state = reservoir.neuron_state_time_matrix.max()
    # shift to positive, by adding the smallest negative state, then divide by the range to scale to [0, 1], then scale to 0..2 and shift to -1..1
    normalized_states = (reservoir.neuron_state_time_matrix - min_state) / (max_state - min_state) * 2 - 1

    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        # shade node colors based on neuron state (values between -1 and 1) RGB: red for positive, blue for negative
        node_colors = [(1.0 - max(0, state), 1.0 - abs(state), 1.0 - max(0, -state)) for state in normalized_states[frame]]
        # adjust edge transparency based on normalized weights RGBA: darker for higher absolute weights
        # edge_colors = [(0, 0, 0, abs(normalized_weights[e.source, e.target])) for e in g.es]
        ig.plot(
            g, 
            target=ax,
            layout=layout,
            vertex_color=node_colors,
            vertex_size=20,
            vertex_label_size=8,
            edge_width=0.5,
            edge_color=[edge["color"] for edge in g.es],
            edge_arrow_size=0.5,
            edge_arrow_width=0.2,
            margin=30
        )
        ax.set_title(f"Reservoir at Timestep {frame}/{time_steps}")
        ax.set_axis_off()
        return ax

    frames = tqdm(range(time_steps), desc="Rendering animation")
    ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=False)
    ani.save(file_name, writer='pillow', fps=10, dpi=100)
    plt.close()

if __name__ == '__main__':
    # from agent import Reservoir
    # reservoir = Reservoir(10)
    # draw_reservoir_graph(reservoir, '0')
    # extract_agent_trajectory('0032', 1, 5)
    sigma = 0.032
    from agent import Reservoir
    reservoir = Reservoir(time_steps=100, num_neurons=1000, burn_in_time=10, mean=0, standard_deviation=sigma)
    plot_activity(reservoir, 'reservoirs', sigma)