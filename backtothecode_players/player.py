from backtothecode_gym.envs.lib.board import ReadOnlyBoard

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0

    def reset(self, id, board : ReadOnlyBoard):
      self.id = id
      self.board = board

    def move(self):
      raise NotImplementedError()