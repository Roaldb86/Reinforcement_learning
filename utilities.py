import numpy as np

def hash_state(state):
    return str(state)

def create_uniform_grid(low, high, bins=(10, 10, 10, 10)):
    """
    Define a uniformly-spaced grid that can be used to discretize a space.

    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension