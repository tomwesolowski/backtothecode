from backtothecode_gym.envs.lib.board import ReadOnlyBoard

class Player:
    def __init__(self):
      pass

    def reset(self, id, board : ReadOnlyBoard):
      self.id = id
      self.board = board

    def move(self):
      raise NotImplementedError()