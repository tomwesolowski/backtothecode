import ipycanvas
import numpy as np

from IPython.display import display

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


class CanvasRenderer(Renderer):
    def __init__(self) -> None:
        super().__init__()
        self.background_color = '#eeeeee'
        self.grid_border_color = '#999'
        self.grid_border_width = 1
        self.cell_pixels = 30
        self.player_to_color = {0: '#eb2a37', 1: '#33b825'}
        self.owner_to_color ={0: '#dba4a4', 1: '#aadba4'}
        self.player_border_color = '#555555'
        self.player_border_width = 2

    def draw_grid(self):
      # with hold_canvas(canvas):
        self.canvas.stroke_style = self.grid_border_color
        self.canvas.line_width = self.grid_border_width
    
        # draw the grid onto the canvas
        for y in np.arange(0, self.grid_height * self.cell_pixels, self.cell_pixels, dtype=float):
            self.canvas.stroke_line(0, y, self.grid_width * self.cell_pixels, y)
        for x in np.arange(0, self.grid_width * self.cell_pixels, self.cell_pixels, dtype=float):   
            self.canvas.stroke_line(x, 0, x, self.grid_height * self.cell_pixels)
    
    def draw_owned_cells(self):
        ownerships = self.board.get_ownerships()
        height, width = ownerships.shape
        for y in range(height):
            for x in range(width):
                if (color := self.owner_to_color.get(ownerships[y][x], None)):
                    self.canvas.fill_style = color
                    self.canvas.fill_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)
    
    def draw_players(self):
        for id, position in enumerate(self.board.get_player_positions()):
            self.canvas.stroke_style = self.player_border_color
            self.canvas.line_width = self.player_border_width
            self.canvas.fill_style = self.player_to_color[id]
            y, x = position
            self.canvas.fill_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)
            self.canvas.stroke_rect(x * self.cell_pixels, y * self.cell_pixels, self.cell_pixels, self.cell_pixels)
    
    def reset(self, board : ReadOnlyBoard):
        super().reset(board)
        self.grid_height = self.board.height
        self.grid_width = self.board.width
        self.canvas = ipycanvas.Canvas(
            width=self.grid_width * self.cell_pixels, 
            height=self.grid_height * self.cell_pixels, 
            sync_image_data=True
        )
        display(self.canvas)
    
    def render(self, round_number):
        with ipycanvas.hold_canvas():
            self.canvas.clear()
            self.canvas.fill_style = self.background_color
            self.canvas.fill_rect(0, 0, self.canvas.width, self.canvas.height)
            self.draw_grid()
            self.draw_owned_cells()
            self.draw_players()