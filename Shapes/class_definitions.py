import numpy as np
from skimage.draw import ellipse, circle, rectangle, polygon



COLORS = {
    'red': np.array([1., 0., 0.]),
    'green': np.array([0, 1., 0]),
    'blue': np.array([0., 0., 1.]),
    'purple': np.array([1., 0., 1.]),
    'yellow': np.array([1., 1., 0.])
}

SHAPES = {
    'ellipse': ellipse,
    'circle': circle,
    'rectangle': rectangle,
    'triangle': polygon,
    'square': rectangle
}