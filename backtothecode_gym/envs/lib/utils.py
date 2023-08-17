def calculate_distance(position, destination):
    py, px = position
    dy, dx = destination
    return abs(px-dx) + abs(py-dy)


def get_directions():
    # WENS. Horizontal directions first
    yield (0, -1) # left  (W)
    yield (0, 1)  # right (E)
    yield (-1, 0) # up    (N)
    yield (1, 0)  # down  (S)


def move_in_direction(position, direction):
    py, px = position
    dy, dx = direction
    return (py+dy, px+dx)
