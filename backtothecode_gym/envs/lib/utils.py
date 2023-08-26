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


def get_all_directions():
    yield from get_directions()
    yield (-1, -1) # (NW)
    yield (-1, 1)  # (NE)
    yield (1, -1)  # (SW)
    yield (1, 1)   # (SE)
    

def get_neighbours(position):
    for direction in get_directions():
        yield move_in_direction(position, direction)


def get_all_neighbours(position):
    for direction in get_all_directions():
        yield move_in_direction(position, direction)


def get_direction_from_action(action):
    return list(get_directions())[action]


def get_action_from_direction(direction):
    return list(get_directions()).index(direction)


def move_in_direction(position, direction):
    py, px = position
    dy, dx = direction
    return (py+dy, px+dx)


def get_opposite_direction(direction):
    y, x = direction
    return (-y, -x)


def get_opposite_action(action):
    direction = get_direction_from_action(action)
    opp_direction = get_opposite_direction(direction)
    return get_action_from_direction(opp_direction)