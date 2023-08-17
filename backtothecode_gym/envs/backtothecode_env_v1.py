import numpy as np

from enum import IntEnum

from .backtothecode_env_v0 import BackToTheCodeEnv_v0

''' simple helper class to enumerate actions in the grid levels '''
class Actions(IntEnum):  
    Stay  = 0    
    North = 1
    East  = 2
    South = 3
    West  = 4

    # get the enum name without the class
    def __str__(self): return self.name  

class BackToTheCodeEnv_v1( BackToTheCodeEnv_v0 ):

  metadata = {'render_modes': ['human']}

  def __init__(self, **kwargs):
      super().__init__(**kwargs)

      # the start and end positions in the grid
      # - by default these are the top-left and bottom-right respectively
      self.start = kwargs.get('start',[0,0])
      self.end = kwargs.get('end',[self.max_x,self.max_y])

      # Baby Robot's initial position
      # - by default this is the grid start
      self.initial_pos = kwargs.get('initial_pos',self.start)

      # Baby Robot's position in the grid
      self.x = self.initial_pos[0]
      self.y = self.initial_pos[1]

      self.last_action = 0
      self.last_reward = 0


  def take_action(self, action):
      ''' apply the supplied action '''
      self.last_action = action

      # move in the direction of the specified action
      if   action == Actions.North: self.y -= 1
      elif action == Actions.South: self.y += 1
      elif action == Actions.West:  self.x -= 1
      elif action == Actions.East:  self.x += 1

      # make sure the move stays on the grid
      if self.x < 0: self.x = 0
      if self.y < 0: self.y = 0
      if self.x > self.max_x: self.x = self.max_x
      if self.y > self.max_y: self.y = self.max_y


  def step(self, action):
      # take the action and update the position
      self.take_action(action)
      obs = np.array([self.x,self.y])

      # set the 'terminated' flag if we've reached the exit
      terminated = (self.x == self.end[0]) and (self.y == self.end[1])
      truncated = False

      # get -1 reward for each step
      # - except at the terminal state which has zero reward
      reward = 0 if terminated else -1
      self.last_reward = reward

      info = {}
      return obs, reward, terminated, truncated, info


  def render(self, **kwargs ):
      print(f"{Actions(self.last_action): <5}: ({self.x},{self.y}) reward = {self.last_reward}")