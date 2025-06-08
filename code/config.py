from pathlib import Path
import multiprocessing as mp

# x, y, direction, boolean whether agent ate 
NUM_MOTION_ATTRIBUTES = 4

PROJECT_ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT_PATH / 'data'

MAX_FOOD_GENERATION_ATTEMPTS = 1000

MAX_PROCESSES = min(mp.cpu_count(), 5)
