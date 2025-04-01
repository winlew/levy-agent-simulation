import numpy as np
import math

def vector_to_angle(normalized_vector):
    """
    Transforms a normalized 2D vector to an angle in radians.
    
    Args:
        normalized_vector (np.array): 2D vector with length 1

    Returns:
        angle (float): angle in radians (ego perspective)
    """
    angle = np.arctan2(normalized_vector[1], normalized_vector[0])
    return angle

def calculate_angle_difference(food_direction, agent_direction):
    """
    Calculates the difference between two angles.
    1. Shift food direction to world perspective by adding pi
    2. Substract the angles
    3. Transform negative differences to positive ones by wrapping around 2pi
    4. Shift back to ego perspective by substracting pi

    Args:
        food_direction (float): angle from agent to food particle in radians (ego perspective)
        agent_direction (float): angle that the agent is facing in radians (world perspective)

    Returns:
        delta (float): angle difference in radians (ego perspective)
    """
    assert(-np.pi <= food_direction <= np.pi)
    assert(0 <= agent_direction <= 2*np.pi)
    delta = (food_direction + np.pi - agent_direction) % (2 * np.pi) - np.pi
    return delta

# TODO refactor and understand. This logic does not work for periodic boundaries
def counter_clockwise(p1, p2, p3):
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return counter_clockwise(A,C,D) != counter_clockwise(B,C,D) and counter_clockwise(A,B,C) != counter_clockwise(A,B,D)

class Point:
    """
    A point in 2D space.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        return Point(self.x * other, self.y * other)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y

class Rectangle:
    """
    Rectangle in 2D space. Defined by its four corner points.
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

def rectangle_from_points(point1, point2, radius):
    """
    Constructs a rectangle object that spans between the two given points.
    The vector between the two points is the midline of the rectangle.
    The sides perpendicular to the midline have length two times the radius. 
    """
    midline = point2 - point1
    perpendicular_vector = Point(-midline.y / math.sqrt(midline.dot(midline)), midline.x / math.sqrt(midline.dot(midline)))
    a = point1 + perpendicular_vector * radius
    b = point1 - perpendicular_vector * radius
    c = point2 - perpendicular_vector * radius
    d = point2 + perpendicular_vector * radius
    return Rectangle(a, b, c, d)

def inside_rectangle(rectangle, point):
    """
    Tests if a point is inside the borders of a rectangle.
    Chooses any corner point as the reference point.
    Then checks whether the projections of the vector from this reference point to the point and the sides of the rectangle are within the borders.
    """
    point_of_reference = rectangle.a
    rectangle_width_vector = rectangle.b - point_of_reference
    rectangle_height_vector = rectangle.d - point_of_reference
    point_difference_vector = point - point_of_reference
    relative_width_projection = rectangle_width_vector.dot(point_difference_vector) / rectangle_width_vector.dot(rectangle_width_vector)
    relative_height_projection = rectangle_height_vector.dot(point_difference_vector) / rectangle_height_vector.dot(rectangle_height_vector)
    if 0 <= relative_width_projection <= 1 and 0 <= relative_height_projection <= 1:
        return True 
    else:
        return False

if __name__ == '__main__':
    pass