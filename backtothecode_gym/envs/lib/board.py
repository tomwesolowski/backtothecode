import numpy as np

from collections import defaultdict

from . import utils

class Board:
    def __init__(self, height, width) -> None:
        self.height, self.width = height, width

    def reset(self, initial_positions):
        self._ownership = np.full((self.height, self.width), -1)
        self.num_players = len(initial_positions)
        self._positions = initial_positions
        for player_id in range(self.num_players):
            self.set_ownership(self.get_position(player_id), player_id) 
        self._actions_buffer = []

    @property
    def shape(self):
        return (self.height, self.width)

    def get_all_cells(self):
         for y in range(self.height):
            for x in range(self.width):
                yield (y, x)

    def get_free_cells(self):
         for position in self.get_all_cells():
             if self.get_ownership(position) == -1:
                 yield position
    
    def within_board(self, position):
        y, x = position
        return (
            y >= 0 and 
            x >= 0 and 
            y < self.height and 
            x < self.width
        )

    def get_position(self, player_id):
        return self._positions[player_id]
    
    def set_position(self, player_id, position):
        self._positions[player_id] = position

    def set_ownership(self, position, player_id):
        y, x = position
        if self._ownership[y][x] != -1:
            raise Exception(f"Field ({y}, {x}) is already owned")
        self._ownership[y][x] = player_id
    
    def get_ownership(self, position):
        y, x = position
        return self._ownership[y][x]
    
    def add_to_buffer(self, player_id, direction_id):
        self._actions_buffer.append((player_id, direction_id))

    def get_possible_actions(self, player_id):
        for action, direction in enumerate(utils.get_directions()):
            new_position = utils.move_in_direction(
                self.get_position(player_id), direction)
            if self.within_board(new_position):
                yield action

    def make_action(self, player_id, action_id):
        direction = list(utils.get_directions())[action_id]
        new_position = self.get_new_position(player_id, direction)
        if not self.within_board(new_position):
            raise Exception(f"Player {player_id} made a forbidden move.")
        self.set_position(player_id, new_position)
    
    def get_new_position(self, player_id, direction):
        position = self.get_position(player_id)
        return utils.move_in_direction(position, direction)
    
    def num_empty_cells(self):
        result = 0
        for y in range(self.height):
            for x in range(self.width):
                result += self.get_ownership((y, x)) == -1
        return result

    def finish_round(self):
        rewards = [0] * self.num_players
        move_failures = [False] * self.num_players
        new_position_count = defaultdict(int)
        # Move players
        for player_id, action_id in self._actions_buffer:
            try:
                self.make_action(player_id, action_id)
            except Exception:
                move_failures[player_id] = True
                # Do not move and continue
        self._actions_buffer.clear()
        # Set ownership of their current cells
        for player_id in range(self.num_players):
            new_position_count[self.get_position(player_id)] += 1
        for player_id in range(self.num_players):
            position = self.get_position(player_id)
            if (self.get_ownership(position) == -1 and 
                new_position_count[position] == 1):
                self.set_ownership(position, player_id)
                rewards[player_id] += 1
        # Set ownership of the surrounded cells
        for player_id in range(self.num_players):
            visited_cells = set()
            for position in self.get_free_cells():
                unclaimed_cells = set()
                if position not in visited_cells and self._get_surrounded_cells(position, visited_cells, unclaimed_cells, player_id):
                    # we did not hit the border nor already onwned cell
                    for position in unclaimed_cells:
                        self.set_ownership(position, player_id)
                        rewards[player_id] += 1
        return move_failures, rewards

    def _get_surrounded_cells(self, position, visited_cells, unclaimed_cells, player_id):
        if not self.within_board(position):
            return False
        current_owner = self.get_ownership(position) 
        if current_owner != -1:
            return current_owner == player_id
        if position in visited_cells:
            return True
        visited_cells.add(position)
        unclaimed_cells.add(position)
        so_far_so_good = True
        for direction in utils.get_all_directions():
            new_position = utils.move_in_direction(position, direction)
            so_far_so_good &= self._get_surrounded_cells(new_position, visited_cells, unclaimed_cells, player_id)
        return so_far_so_good

    def _find_best_direction(self, position, destination):
        best_direction, best_distance = None, 1e9
        for direction in utils.get_directions():
            new_position = utils.move_in_direction(position, direction)
            if not self._within_board(new_position):
                continue
            distance = utils.calculate_distance(
                new_position, destination
            )
            if distance < best_distance:
                best_distance, best_direction = distance, direction
        return best_direction
            

class ReadOnlyBoard:
    def __init__(self, board):
        self._board = board

    def get_position(self, player_id):
        return self._board.get_position(player_id)
    
    def get_ownership(self, position):
        return self._board.get_ownership(position)

    def get_ownerships(self):
        return self._board._ownership
    
    def within_board(self, position):
        return self._board.within_board(position)
    
    def get_possible_actions(self, player_id):
        return self._board.get_possible_actions(player_id)

    def get_player_positions(self):
        return (self.get_position(player_id) for player_id in range(self.num_players))
    
    @property
    def height(self):
        return self._board.height
    
    @property
    def width(self):
        return self._board.width
    
    @property
    def num_players(self):
        return self._board.num_players
    
    @property
    def num_empty_cells(self):
        return self._board.num_empty_cells

    @property
    def shape(self):
        return self._board.shape