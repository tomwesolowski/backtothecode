import ipycanvas
import numpy as np

from IPython.display import display

from .board import ReadOnlyBoard

class Renderer:
    def __init__(self) -> None:
        pass

    def render(self, board, players, round_number):
        raise NotImplementedError()
    

class NoRenderer:
    def render(self, board, players, round_number):
        pass


class PrintRenderer(Renderer):
  def __init__(self) -> None:
    super().__init__()
    self.player_marks = ['O', 'X']
    self.empty_mark = '.'

  def render(self, board, players, round_number):
    print(f"------- Round {round_number}:")
    for player in players:
        y, x = board.get_position(player.id)
        print(f"Player {player.name}: ({y}, {x})")
    owner_to_mark = {
        -1: self.empty_mark,
    }
    position_to_player = {
        board.get_position(id): id for id in range(board.num_players)
    }
    for id, mark in enumerate(self.player_marks):
        owner_to_mark[id] = mark.lower()
    for y in range(board.height):
        for x in range(board.width):
            if (y, x) in position_to_player:
                player_id = position_to_player[(y, x)]
                print(self.player_marks[player_id], end='')
            else:
                owner_id = board.get_ownership((y,x))
                print(owner_to_mark[owner_id], end='')
        print('')


class CanvasRenderer(Renderer):
    def __init__(self) -> None:
        super().__init__()
        self.background_color = '#eeeeee'
        self.grid_border_color = '#999'
        self.grid_border_width = 1
        self.cell_pixels = 30
        self.player_to_color = {1: '#eb2a37', 0: '#33b825'}
        self.owner_to_color ={1: '#dba4a4', 0: '#aadba4'}
        self.player_border_color = '#555555'
        self.player_border_width = 2
        self.font_size = 32

    def draw_grid(self):
      # with hold_canvas(canvas):
        self.canvas.stroke_style = self.grid_border_color
        self.canvas.line_width = self.grid_border_width
    
        # draw the grid onto the canvas
        for y in np.arange(0, self.grid_height * self.cell_pixels, self.cell_pixels, dtype=float):
            self.canvas.stroke_line(0, y, self.grid_width * self.cell_pixels, y)
        for x in np.arange(0, self.grid_width * self.cell_pixels, self.cell_pixels, dtype=float):   
            self.canvas.stroke_line(x, 0, x, self.grid_height * self.cell_pixels)
    
    def draw_owned_cells(self, board):
        ownerships = board.get_ownerships()
        height, width = ownerships.shape
        for y in range(height):
            for x in range(width):
                if (color := self.owner_to_color.get(ownerships[y][x], None)):
                    self.canvas.fill_style = color
                    self.canvas.fill_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)
    
    def draw_players(self, board):
        for id, position in enumerate(board.get_player_positions()):
            self.canvas.stroke_style = self.player_border_color
            self.canvas.line_width = self.player_border_width
            self.canvas.fill_style = self.player_to_color[id]
            y, x = position
            self.canvas.fill_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)
            self.canvas.stroke_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)

    def draw_scores(self, players):
        for player in players:
            self.canvas.font = f"{self.font_size}px serif"
            self.canvas.stroke_style = self.player_to_color[player.id]
            self.canvas.stroke_text(f"{player.name}: {player.score}", 
                               x=0, y=self.grid_height * self.cell_pixels + (player.id + 1) * self.font_size)
    
    def draw_canvas(self, board):
        self.grid_height = board.height
        self.grid_width = board.width
        self.canvas = ipycanvas.Canvas(
            width=self.grid_width * self.cell_pixels, 
            height=self.grid_height * self.cell_pixels + (len(self.player_to_color) + 1) * self.font_size, 
            sync_image_data=True
        )
        display(self.canvas)
    
    def render(self, board, players, round_number):
        if not hasattr(self, 'canvas'):
            self.draw_canvas(board)
        with ipycanvas.hold_canvas():
            self.canvas.clear()
            self.canvas.fill_style = self.background_color
            self.canvas.fill_rect(0, 0, self.canvas.width, self.canvas.height)
            self.draw_grid()
            self.draw_owned_cells(board)
            self.draw_players(board)
            self.draw_scores(players)