import numpy as np
import config
# hide pygame support prompt
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class Environment:
    """
    A quadratic 2D environment with food particles and periodic boundary conditions.
    - generates the positions of the food particles
    - finds the closest food particle to an agent
    """

    def __init__(self, params):
        """
        Args:
            params (Params): class that stores parameters of the simulation
                size (int): size of the environment
                num_food (int): number of food particles
                border_buffer (float): minimum distance between food particles and the border of the environment
                food_buffer (float): minimum distance between food particles
                seed (int): random seed for reproducibility of random number generation
        """
        self.size = params.size
        self.num_food = params.num_food
        self.rng = np.random.default_rng(params.seed)
        if params.empty:
            self.food_positions = np.asarray([])
        else:
            self.food_positions = self.generate_food_positions(params.border_buffer, params.food_buffer)
        
        self.walls = []
    
    def add_wall(self, start, end):
        """
        Add a wall defined by start and end points.
        
        Args:
            start (np.array): Starting point of the wall (2D).
            end (np.array): Ending point of the wall (2D).
        """
        self.walls.append((start, end))

    def generate_food_positions(self, border_buffer, food_buffer):
        """
        Randomly generate positions for the food particles.
        But enforce a minimum distance between them and the border of the environment.

        Args:
            border_buffer (float): minimum distance between food particles and the border of the environment
            food_buffer (float): minimum distance between food particles

        Returns:
            food_positions (np.array): 2D array with the x and y positions of the food particles

        Raises:
            ValueError: if the maximum number of attempts to generate food positions is reached
        """
        food_count = 0
        food_positions = np.empty((0, 2))
        failed_attempts = 0
        max_attempts = config.MAX_FOOD_GENERATION_ATTEMPTS

        while food_count < self.num_food:   
            suggested_food_x_position = self.rng.uniform(border_buffer, self.size - border_buffer)
            suggested_food_y_position = self.rng.uniform(border_buffer, self.size - border_buffer)

            position_valid = True

            # check whether distance to all other food particles is large enough
            for food_position in food_positions:
                distance = np.linalg.norm([suggested_food_x_position - food_position[0], suggested_food_y_position - food_position[1]])
                if distance < food_buffer:
                    position_valid = False
                    failed_attempts += 1
                    break
            
            if failed_attempts > max_attempts:
                raise ValueError("Failed to generate food positions.")
            
            if position_valid:
                food_positions = np.vstack([food_positions, [suggested_food_x_position, suggested_food_y_position]])
                food_count += 1

        return food_positions

    def get_closest_food(self, agent):
        """
        Find the closest food particle to the agent within its perception radius.

        Args:
            agent (Agent): the agent that is searching for food

        Returns:
            closest_food_position (np.array): 2D array with the x and y position of the closest food particle
            closest_distance (float): distance to the closest food particle
            closest_index (int): index of the closest food particle in the food_positions array
        """

        # blind agent does not see anything
        # eat radius matters as well!
        if self.food_positions.size == 0:
            return None, None, None

        dx = np.abs(self.food_positions[:, 0] - agent.position[0])
        dy = np.abs(self.food_positions[:, 1] - agent.position[1])
        # correct for periodic boundaries
        dx = np.minimum(dx, self.size - dx)
        dy = np.minimum(dy, self.size - dy)

        distances = np.sqrt(dx**2 + dy**2)

        # filter for visible and unconsumed food
        in_range = (distances <= agent.perception_radius) & (~agent.food_mask)
        if distances[in_range].size > 0:
            closest_distance = np.min(distances[in_range])
            closest_index = np.where(distances == closest_distance)[0][0]
            closest_position = self.food_positions[closest_index]
        else:
            closest_distance = None
            closest_index = None
            closest_position = None

        return closest_position, closest_distance, closest_index

    def custom_food_positioning(self):
        """
        Custom food positioning using a Pygame window.
        """
        self.food_positions = np.empty((0, 2))
        self.num_food = 0
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_size = min(screen_width, screen_height) // 2
        pygame.init()
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Click to position food particles")
        scale_factor = window_size / self.size
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = pygame.mouse.get_pos()
                    self.food_positions = np.vstack([self.food_positions, [position[0] / scale_factor, position[1] / scale_factor]])
            screen.fill((255, 255, 255))
            for food_position in self.food_positions:
                scaled_position = (int(food_position[0] * scale_factor), int(food_position[1] * scale_factor))
                pygame.draw.circle(screen, (0, 0, 0), scaled_position, 5)
            pygame.draw.circle(screen, (255, 0, 0), (int(self.size / 2 * scale_factor), int(self.size / 2 * scale_factor)), 5)
            pygame.display.flip()   
        pygame.quit()
        self.num_food = len(self.food_positions)


if __name__ == '__main__':
    from visualization import visualize_state
    from parameters import Params
    import matplotlib.pyplot as plt

    params = Params.from_json('parameters.json')
    env = Environment(params)
    ax = visualize_state(env, None)
    plt.show()