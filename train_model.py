import random, csv, copy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as step
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common
from tf_agents.metrics import py_metric
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver

CORRECT_PROB = 0.5 # the probability our agent gets the question correct
MODEL_STRUCTURE = (512, 256, 32)
BATCH_SIZE = 32
BUFFER_SIZE = 8192
TRAIN_EPISODES = 1
NUM_ITERATIONS = 10000
LOG_INTERVAL = 1000 
SAVE_INTERVAL = 1000
LEARNING_RATE = 1e-4
LAMBDA =  1e-8
DATA_FILEPATH = "/content/Final_J_data.csv"

class Jeopardy(py_environment.PyEnvironment):
  def __init__(self, filepath):
    self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0.0, maximum=25.0)
    self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=0.0, maximum=100)
    self._state = None
    self._hidden_info = None
    self._episode_end = False
    self.data = None
    self.data_filepath = filepath
    self.data_generator = self.init_data_reader()

  def action_spec(self):
      return self._action_spec

  def observation_spec(self):
      return self._observation_spec

  def _reset(self):
    self._state = np.array(self._orig_state)
    self._hidden_info = np.array(self._orig_hidden_info)
    self._episode_end = False
    self._current_time_step = step.restart(np.array(self._state))
    return self._current_time_step

  def _step(self, action, simple_reward=False):
    if self._episode_end:
      self.reset()

    # update the agent's score
    if action < 0.0:#> self._state[0]:
      print(action)
      self._episode_end = True
      reward = -100.0
      return step.termination(np.array(self._state, dtype=np.float32), reward)
      
    action = min(max(0, action), self._state[0])
    
    correct = random.random() < CORRECT_PROB
    change = action if correct else -action

    self._state[0] += change

    # update the other two using the information given
    self._state[1:3] = self._hidden_info[1:]

    self._episode_end = True
    
    # reward is distributed as follows:
    # -1.0 for 3rd place
    # -0.5 for 2nd place
    # 1.0 for 1st place
    # 0.5 if tied for 1st
    if max(self._state[:3]) == self._state[0]:
      if self._state[1] == self._state[0] or self._state[2] == self._state[0]:
        # tied for 1st
        if simple_reward:
          reward = 1.0
        else:
          reward = 20.0
      else:
        # absolute 1st
        if simple_reward:
          reward = 1.0
        else:
          reward = min(20.0, self._state[0])
    elif min(self._state[:3]) == self._state[0]:
      # 3rd
      if simple_reward:
          reward = 0.0
      else:
        reward = 1.0
    else:
      # 2nd
      if simple_reward:
          reward = 0.0
      else:
        reward = 2.0
    
    return step.termination(np.array(self._state, dtype=np.float32), reward)

  def init_data_reader(self):
    # create a csv data reader
    #if self.data is not None:
      #self.data.close()
    self.data = None
    self.data = open(self.data_filepath)
    data_generator = csv.reader(self.data)
    return data_generator
  
  def next_datum(self):
    state, hidden = [], []# if data is None else data[0], data[1]
    while len(state) != 3 or len(hidden) != 3:
      try:
        state, hidden = next(self.data_generator)
      except StopIteration:
        self.data_generator = self.init_data_reader()
        continue

      # take csv data (in the form of strings) and split it into lists of state data
      # then, convert to an np array and scale down by 1000 to make the numbers more managable
      state = state[1:-1].split(', ')
      #state.append(CORRECT_PROB*1000)
      state, hidden = np.array(state, dtype=np.float32)/1000, np.array(hidden[1:-1].split(', '), dtype=np.float32)/1000

    # scramble the order so the computer isn't always the rightmost player
    #rand_idxs = [0, 1, 2]
    #random.shuffle(rand_idxs)
    self._state = np.array(state, dtype=np.float32)#np.array([state[rand_idxs[0]], state[rand_idxs[1]], state[rand_idxs[2]], state[3]], dtype=np.float32)
    self._hidden_info = np.array(hidden, dtype=np.float32)#np.array([hidden[rand_idxs[0]], hidden[rand_idxs[1]], hidden[rand_idxs[2]]], dtype=np.float32)
    self._orig_state = np.array(state, dtype=np.float32)
    self._orig_hidden_info = np.array(hidden, dtype=np.float32)

  def setup_custom(self, init, after):
    # set up a custom environment
    self._orig_state = np.array(init, dtype=np.float32)
    self._orig_hidden_info = np.array(after, dtype=np.float32)
    return self.reset()

python_env = Jeopardy(DATA_FILEPATH)
env = tf_py_environment.TFPyEnvironment(python_env)

optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)

training_steps_complete = tf.compat.v2.Variable(0)

actor = actor_distribution_network.ActorDistributionNetwork(env.observation_spec(), env.action_spec(), fc_layer_params=MODEL_STRUCTURE)
value_net = value_network.ValueNetwork(env.observation_spec())
agent = reinforce_agent.ReinforceAgent(env.time_step_spec(), env.action_spec(), actor_network=actor, value_network=value_net, optimizer=optimizer, entropy_regularization=LAMBDA, train_step_counter=training_steps_complete)
agent.initialize()
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

policy_hist = []

buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec, env.batch_size, BUFFER_SIZE)

all_reward = []
loss = []

def play_episodes(env, policy, num_episodes, verbose=False):
  env.pyenv._envs[0].next_datum()
  env.reset()
  episode = 0
  tot_reward = 0
  while episode < num_episodes:
    curr_ts = env.current_time_step()
    action_step = policy.action(curr_ts)
    next_ts = env.step(action_step.action)
    traj = trajectory.from_transition(curr_ts, action_step, next_ts)
    if verbose:
          print(f"Current state: {curr_ts}\nAction taken: {action_step}\n New state: {next_ts}\n======================")

    tot_reward += curr_ts.reward
    
    buffer.add_batch(traj)

    if traj.is_boundary():
      # if the episode is complete, increase the counter
      episode += 1
  
  return tot_reward

for iter in range(NUM_ITERATIONS):
  all_reward.append(play_episodes(env, agent.collect_policy, TRAIN_EPISODES)[0]/TRAIN_EPISODES)
  experiences = buffer.gather_all()
  recent_loss = agent.train(experiences).loss
  loss.append(recent_loss)
  buffer.clear()

  steps_complete = agent.train_step_counter.numpy()
  if steps_complete % SAVE_INTERVAL == 0:
    policy_hist.append(copy.deepcopy(agent.policy))
  if steps_complete % LOG_INTERVAL == 0:
    print(f"\r{steps_complete} steps complete; Loss = {recent_loss:.3f}")

  if steps_complete % 25 == 0:
    print(f"\r{steps_complete} steps complete; Loss = {recent_loss:.3f}", end='')

# remove outliers
arr = np.array(loss)
z = (arr - np.mean(arr)) / np.std(arr)
arr = np.where(abs(z) < 3, arr, None)

# plot the new data
plt.figure()

plt.plot(loss)

plt.show()

# evaluate the policy
def evaluate_policies(loops, *policies):
  rewards = np.zeros((len(policies), loops))
  win_rates = np.zeros(len(policies))
  for i in range(loops):
    env.pyenv._envs[0].next_datum()
    env.reset()

    curr_ts = env.current_time_step()
    for num, policy in enumerate(policies):
      if isinstance(policy, float):
        action_step = policy_step.PolicyStep(policy, (), ())
      elif isinstance(policy, str):
        action_step = policy_step.PolicyStep(float(abs(env.pyenv._envs[0]._hidden_info[0] - env.pyenv._envs[0]._state[0])))
      else:
        action_step = policy.action(curr_ts)
      ts = env.pyenv._envs[0]._step(action_step.action, simple_reward=True)
      env.reset()
      rewards[num, i] = ts.reward

  return rewards

best_policy = policy_hist[-1]
NUM_TRIALS = 1000
results = evaluate_policies(NUM_TRIALS, best_policy, 'human')
results = np.mean(results, axis=1)
print(f"Win percentages based on {NUM_TRIALS} trials\n=========================\nHuman: {results[1]*100:.1f}%\nActor: {results[0]*100:.1f}%\n=========================")