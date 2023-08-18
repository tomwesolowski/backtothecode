from .board import ReadOnlyBoard

class Renderer:
  def __init__(self) -> None:
    pass

  def reset(self, board : ReadOnlyBoard):
      self.board = board

  def render(self, round_number):
    raise NotImplementedError()


class PrintRenderer(Renderer):
  def __init__(self) -> None:
    super().__init__()
    self.player_marks = ['O', 'X']
    self.empty_mark = '.'

  def render(self, round_number):
    print(f"------- Round {round_number}:")
    for id in range(self.board.num_players):
        y, x = self.board.get_position(id)
        print(f"Player {id}: ({y}, {x})")
    owner_to_mark = {
        -1: self.empty_mark,
    }
    position_to_player = {
        self.board.get_position(id): id for id in range(self.board.num_players)
    }
    for id, mark in enumerate(self.player_marks):
        owner_to_mark[id] = mark.lower()
    for y in range(self.board.height):
        for x in range(self.board.width):
            if (y, x) in position_to_player:
                player_id = position_to_player[(y, x)]
                print(self.player_marks[player_id], end='')
            else:
                owner_id = self.board.get_ownership((y,x))
                print(owner_to_mark[owner_id], end='')
        print('')