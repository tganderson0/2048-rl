import numpy as np

class Buffer:
  def __init__(self, state_space, action_space, buffer_capacity=100_000, batch_size=64):
    self.buffer_capacity = buffer_capacity
    self.batch_size = batch_size
    self.buffer_counter = 0

    self.state_buffer = np.zeros((self.buffer_capacity, state_space))
    self.action_buffer = np.zeros((self.buffer_capacity, action_space))
    self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    self.next_state_buffer = np.zeros((self.buffer_capacity, state_space))
  
  """
  Takes a (state, action, reward, state')
  Saves the values to the buffer
  """
  def record(self, obs_tuple):
    index = self.buffer_counter % self.buffer_capacity

    self.state_buffer[index] = obs_tuple[0]
    self.action_buffer[index] = obs_tuple[1]
    self.reward_buffer[index] = obs_tuple[2]
    self.next_state_buffer[index] = obs_tuple[3]

    self.buffer_counter += 1
