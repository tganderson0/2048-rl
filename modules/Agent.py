from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from .Buffer import Buffer
from .OUActionNoise import OUActionNoise
from tqdm import tqdm

import os

# 0 = left, 1 = up, 2 = right, 3 = down


class Agent:
  def __init__(self):
    self.state_space = 16
    self.action_space = 4

    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.gamma = 0.99
    self.batch_size = 128

    self.tau = 0.005
    self.epochs = 10_000
    self.std_dev = 0.2
    self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
    self.critic_learning_rate = 0.002
    self.actor_learning_rate = 0.001
    self.learns = 0
    self.actor_model = self.create_actor()
    self.critic_model = self.create_critic()
    self.actor_target_model = self.create_actor()
    self.critic_target_model = self.create_critic()
    self.rewards = []
    self.buffer = Buffer(self.state_space, self.action_space, batch_size=self.batch_size)

    self.critic_optimizer = Adam(self.critic_learning_rate)
    self.actor_optimizer = Adam(self.actor_learning_rate)

    self.prev_action = 0

    self.allowed_invalid_moves = 10

  def create_actor(self):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = Input(shape=(self.state_space,))
    out = Dense(256, activation='relu')(inputs)
    out = Dense(256, activation='relu')(out)

    outputs = Dense(self.action_space, activation='tanh', kernel_initializer=last_init)(out)

    model = Model(inputs, outputs)
    return model

  def create_critic(self):
    state_input = Input(shape=(self.state_space,))
    state_out = Dense(16, activation='relu')(state_input)
    state_out = Dense(32, activation='relu')(state_out)

    action_input = Input(shape=(self.action_space,))
    action_out = Dense(32, activation='relu')(action_input)

    concat_layer = Concatenate()([state_out, action_out])

    out = Dense(256, activation='relu')(concat_layer)
    out = Dense(512, activation='relu')(out)
    out = Dense(1024, activation='relu')(out)
    outputs = Dense(1)(out)

    model = Model([state_input, action_input], outputs)
    return model

  @tf.function
  def update(self, state_batch, action_batch, reward_batch, next_state_batch):
    with tf.GradientTape() as tape:
      target_actions = self.actor_target_model(next_state_batch, training=True)
      y = reward_batch + self.gamma * self.critic_target_model([next_state_batch, target_actions], training=True)
      critic_value = self.critic_model([state_batch, action_batch], training=True)
      critic_loss = tf.math.reduce_mean(tf.math.square( y - critic_value))

    critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
    self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
  
  def learn(self):

    record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

    batch_indices = np.random.choice(record_range, self.batch_size)

    state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)

    next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

    self.update(state_batch, action_batch, reward_batch, next_state_batch)

  @tf.function
  def update_target(self, target_weights, weights, tau):
    # Lerp all the values
    for (a, b) in zip(target_weights, weights):
      a.assign(b * tau + a * (1 - tau))
    
  def policy(self, state, noise_object):
    sampled_actions = tf.squeeze(self.actor_model(state))
    noise = noise_object()

    sampled_actions = sampled_actions.numpy() + noise

    legal_action = sampled_actions.argmax()

    legal_action += 1 # since actions are 1-4
    
    self.prev_action = legal_action

    return legal_action

  def train(self, env, load_weights, render=False, train=True):

    ep_reward_list = []
    avg_reward_list = []

    if (load_weights):
      self.actor_model.load_weights("weights/actor.h5")
      self.critic_model.load_weights("weights/critic.h5")
      self.actor_target_model.load_weights("weights/actor_target.h5")
      self.critic_target_model.load_weights("weights/critic_target.h5")
    
    for i in tqdm(range(self.epochs)):
      
      prev_state = env.reset()

      self.prev_action = 0

      prev_state = prev_state.flatten()

      episodic_reward = 0

      invalid_move_count = self.allowed_invalid_moves

      while True:
        
        if (render):
          env.render()

        # os.system('cls||clear')
        
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = self.policy(tf_prev_state, self.ou_noise)

        # tqdm.write(f"Action: {action}")

        state, reward, done, info = env.step(action)

        if (reward != 0):
          invalid_move_count = self.allowed_invalid_moves
        if (reward == 0):
          invalid_move_count -= 1

        state = state.flatten()

        if train:
          self.buffer.record((prev_state, action, reward, state))
          episodic_reward += reward

          self.learn()
          self.update_target(self.actor_target_model.variables, self.actor_model.variables, self.tau)
          self.update_target(self.critic_target_model.variables, self.critic_model.variables, self.tau)
      
        if done or invalid_move_count == 0:
          break

        prev_state = state
      
      ep_reward_list.append(episodic_reward)

      avg_reward = np.mean(ep_reward_list[-40:])
      tqdm.write(f"Episode * {i} * Avg Reward: {avg_reward}")
      avg_reward_list.append(avg_reward)

      if i % 100 == 0:
        self.save_weights()
    
    self.save_weights()

    return avg_reward_list


  def save_weights(self):
    self.actor_model.save_weights("weights/actor.h5")
    self.critic_model.save_weights("weights/critic.h5")
    self.actor_target_model.save_weights("weights/actor_target.h5")
    self.critic_target_model.save_weights("weights/critic_target.h5")
  
