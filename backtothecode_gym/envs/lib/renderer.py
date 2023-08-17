from .board import ReadOnlyBoard

class Renderer:
    def __init__(self, board) -> None:
        self.board = board


class PrintRenderer(Renderer):
    def __init__(self, board : ReadOnlyBoard, player_marks, ownership_marks, empty_mark='.') -> None:
        super().__init__(board)
        self.player_marks = player_marks
        self.ownership_marks = ownership_marks
        self.empty_mark = empty_mark

    def render(self):
        ownership_to_mark = {
            -1: self.empty_mark,
        }
        for id, mark in enumerate(self.ownership_marks):
            ownership_to_mark[id] = mark
        for y in range(self.board.height):
            for x in range(self.board.width):
                found_player = None
                for id in range(self.board.num_players):
                    if self.board.get_position(id) == (y, x):
                        found_player = id
                        break
                if found_player:
                    print(self.player_marks[found_player], end='')
                else:
                    print(ownership_to_mark[self.board.get_ownership((y,x))], end='')
            print('')