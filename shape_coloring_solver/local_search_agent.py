import math
import random
import numpy as np
from gridgame import *

##############################################################################################################################

# Visualize the code by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.
# The gs argument controls the grid size.

##############################################################################################################################

game = ShapePlacementGrid(GUI=False, render_delay_sec=0.5, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization
# shapePos is the current position of the brush.
# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py).
# currentColorIndex is the index of the current color being placed (order specified in gridgame.py).

# grid represents the current state of the board. 
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)
# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)

def hill_climbing(game):
    "Use first choice local search to place colored shapes"
    # Setup variables
    current = heuristic_value(game)
    T = 10
    d = 0.90
    no_improvement = 0
    last_cell = None

    while not game.checkGrid(game.grid):
        
        # If there's no improvement for 30 moves, then reset the whole board
        if no_improvement >= 30:
            for _ in range(len(game.placedShapes)):
                game.execute("undo")
            current = heuristic_value(game)
        elif no_improvement >= 15 and no_improvement < 30:  # If there's no improvement for 15 moves, then half reset the board
            for _ in range(len(game.placedShapes) // 2):
                game.execute("undo")
            current = heuristic_value(game)

        # Generate next move
        new_cell = generate_move(game, last_cell)
        if new_cell is None:
            game.execute("undo")
            
        neighbor = heuristic_value(game)

        # Check if eighboring state is better that the current state or simulated annealing accepts it
        if neighbor > current or random.random() < math.exp((neighbor - current) / T):
            current = neighbor
            last_cell = new_cell
            no_improvement = 0
        else:
            # Undo the worse move
            game.execute("undo")
            no_improvement += 1

        # Lower temperature for simulated annealling
        T *= d

def generate_move(game, last_cell=None):
    """Returns a valid move with a random cell, shape, and color.
    None if a valid move can't be found in gridSize^2 tries."""
    tries = 0

    # Try gridSize^2 times to find a valid move, otherwise might have to backtrack
    while tries < game.gridSize ** 2:
        # Pick a random empty cell
        empty_cells = np.argwhere(game.grid == -1)
        if len(empty_cells) == 0:
            return None
        
        # Bias towards cells closer to current brush position
        if last_cell is not None and random.random() < 0.7:
            candidates = [c for c in empty_cells if np.linalg.norm(c - last_cell) < 3]
            if candidates:
                random_cell = np.random.choice(len(candidates))
                cell_pos = [candidates[random_cell][1], candidates[random_cell][0]]
            else:
                random_cell = np.random.choice(len(empty_cells))
                cell_pos = [empty_cells[random_cell][1], empty_cells[random_cell][0]]
        else:
            random_cell = np.random.choice(len(empty_cells))
            cell_pos = [empty_cells[random_cell][1], empty_cells[random_cell][0]]

        # Filter for shapes that fit
        valid_shapes = [s for s in range(len(game.shapes)) if game.canPlace(game.grid, game.shapes[s], cell_pos)]
        if not valid_shapes:
            tries += 1
            continue

        # Pick a random shape (by its index)
        shape = np.random.choice(valid_shapes)

        # Pick a random color (by its index)
        color = game.getAvailableColor(game.grid, cell_pos[0], cell_pos[1])

        # Move brush and switch the shape/color
        move_to_cell(game, cell_pos)
        switch_shape(game, shape)
        switch_color(game, color)
        
        # Check that this configuration is a valid move
        if game.canPlace(game.grid, game.shapes[shape], cell_pos):
            grid_copy = game.grid.copy()
            for i, row in enumerate(game.shapes[shape]):
                for j, cell in enumerate(row):
                    if cell:
                        grid_copy[cell_pos[1] + i, cell_pos[0] + j] = color
            if check_adjency(grid_copy):
                game.execute("place")
                return cell_pos
            else:
                tries += 1
    return None

def check_adjency(grid):
    """Check that no adjacent cells have the same color in a given grid"""
    grid_size = grid.shape[0]
    for i in range(grid_size):
        for j in range(grid_size):
            color = grid[i, j]
            if color == -1:
                continue
            if i > 0 and grid[i - 1, j] == color:
                return False
            if i < grid_size - 1 and grid[i + 1, j] == color:
                return False
            if j > 0 and grid[i, j - 1] == color:
                return False
            if j < grid_size - 1 and grid[i, j + 1] == color:
                return False
    return True

def move_to_cell(game, new_cell):
    """Moves the brush to the specified destination cell"""
    c1, r1 = game.shapePos[0], game.shapePos[1]
    c2, r2 = new_cell[0], new_cell[1]

    # Check if any positions are out of bounds
    out_of_bounds = any(x < 0 and x > game.gridSize for x in (c1, r1, c2, r2))
    if out_of_bounds:
        raise ValueError("Coordinate out of bounds!")

    # Move vertically
    while r1 < r2:
        game.execute("down")
        r1 += 1
    while r1 > r2:
        game.execute("up")
        r1 -= 1

    # Move horizontally
    while c1 < c2:
        game.execute("right")
        c1 += 1
    while c1 > c2:
        game.execute("left")
        c1 -= 1

def switch_color(game, new_color):
    """Switches the current color to the specified color"""
    # Check for valid color
    if new_color not in game.colorIdxToName.keys():
        raise ValueError("Not a valid color index!")

    # Switch brush color
    current = game.currentColorIndex
    if new_color == current:
        return
    elif new_color < current:
        offset = len(game.colors) - current
        for _ in range(offset + new_color):
            game.execute("switchcolor")
    else:
        difference = new_color - current
        for _ in range(difference):
            game.execute("switchcolor")

def switch_shape(game, new_shape):
    "Switches the current shape to the specified shape"
    if new_shape not in game.shapesIdxToName.keys():
        raise ValueError("Not a valid shape index!")

    # Switch brush shape
    current = game.currentShapeIndex
    if new_shape == current:
        return
    elif new_shape < current:
        offset = len(game.shapes) - current
        for _ in range(offset +  new_shape):
            game.execute("switchshape")
    else:
        difference = new_shape - current
        for _ in range(difference):
            game.execute("switchshape")

def heuristic_value(game):
    """Computes a heuristic value based 
    on the number of empty cells,
    number of shapes placed,
    and unique colors used"""
    grid = game.grid.copy()
    placed_shapes = game.placedShapes.copy()
    empty_cells = np.sum(grid == -1)
    num_shapes = len(placed_shapes)
    num_colors = len(np.unique(grid)) - 1 if empty_cells > 0 else len(np.unique(grid))
    
    # Count number of cells with adjacent color conflicts
    adj_conflicts = 0
    for x in range(grid.shape[1]):
        for y in range(grid.shape[0]):
            adjacent_colors = set()
            if x > 0:
                adjacent_colors.add(grid[y, x - 1])
            if x < game.gridSize - 1:
                adjacent_colors.add(grid[y, x + 1])
            if y > 0:
                adjacent_colors.add(grid[y - 1, x])
            if y < game.gridSize - 1:
                adjacent_colors.add(grid[y + 1, x])
            if grid[y, x] in adjacent_colors:
                adj_conflicts += 1


    # The smaller (more negative) the number, the worse the move is
    return (-1 / 100)* (100 * adj_conflicts + 50 * empty_cells + 20 * num_shapes + 10 * num_colors)

hill_climbing(game)

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))