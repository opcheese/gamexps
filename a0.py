
import collections
from copyreg import pickle
import datetime
import functools
import itertools
import json
import os
import random
import re
import sys
import tempfile
import time
import traceback
from torch.utils.data import Dataset
import torch
import numpy as np
import simple_model
import model
import mcts
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats
from evaluator import AlphaZeroEvaluator
import main
import math


# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


class TrajectoryState(object):
  """A particular point along a trajectory."""

  def __init__(self, observation, current_player, legals_mask, action, best_action, policy,
               value):
    self.observation = observation
    self.current_player = current_player
    self.legals_mask = legals_mask
    self.action = action
    self.best_action = best_action
    self.policy = policy
    self.value = value

  def __str__(self) -> str:
      return "current_player:{}, action:{},policy:{},value:{} ".format(self.current_player, self.action, self.policy, self.value)


class Trajectory(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self):
    self.states = []
    self.returns = None

  def add(self, information_state, action, best_action, policy):
    self.states.append((information_state, action, best_action, policy))


class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def save(self, path):
    pass
    #a_file = open(path, "wb")
    #pickle.dump(self.data, a_file)

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)


class CustomGameDataset(Dataset):
    def __init__(self, buffer):
        data = buffer.data

        nusize1 = (len(data*(4+1)),) + data[0].observation.shape
        nusize2 = (len(data*(4+1)),) + data[0].legals_mask.shape
        nusize3 = (len(data*(4+1)),) + data[0].policy.shape
        nusize4 = (len(data*(4+1)),)

        self.boards = torch.zeros(
            nusize1, dtype=torch.float32, device=torch.device('cuda:0'))
        self.masks = torch.zeros(
            nusize2, dtype=torch.float32, device=torch.device('cuda:0'))
        self.policy = torch.zeros(
            nusize3, dtype=torch.float32, device=torch.device('cuda:0'))
        self.value = torch.zeros(
            nusize4, dtype=torch.float32, device=torch.device('cuda:0'))
        i = -1
        for _, dat in enumerate(buffer.data):
             i += 1
             augmentation_obs = []
             augmentation_mask = []
             augmentation_policy = []

             #for width=height
             side = len(dat.legals_mask)
             msk = np.reshape(np.copy(dat.legals_mask), (int(math.sqrt(side)), -1))
             pol = np.reshape(np.copy(dat.policy), (int(math.sqrt(side)), -1))

             for rev in range(1, 4):
               obs = dat.observation
               b = np.rot90(obs, rev, (1, 2))
               m = np.rot90(msk, rev)
               p = np.rot90(pol, rev)
               augmentation_obs.append(np.copy(b))
               augmentation_mask.append(np.copy(np.reshape(m, (-1))))
               augmentation_policy.append(np.copy(np.reshape(p, (-1))))
             obs = dat.observation

             flip1_obs = np.flip(obs, 2)
             flip1_m = np.flip(msk, 1)
             flip1_p = np.flip(pol, 1)
             augmentation_obs.append(np.copy(flip1_obs))
             augmentation_mask.append(np.copy(np.reshape(flip1_m, (-1))))
             augmentation_policy.append(np.copy(np.reshape(flip1_p, (-1))))

             self.boards[i] = torch.from_numpy(
                 dat.observation).type(torch.cuda.FloatTensor).cuda()
             self.masks[i] = torch.from_numpy(dat.legals_mask).type(
                 torch.cuda.FloatTensor).cuda()
             self.policy[i] = torch.from_numpy(
                 dat.policy).type(torch.cuda.FloatTensor).cuda()
             self.value[i] = torch.as_tensor(dat.value).type(
                 torch.cuda.FloatTensor).cuda()

             for k in range(len(augmentation_policy)):
              i += 1

              self.boards[i] = torch.from_numpy(augmentation_obs[k]).type(torch.cuda.FloatTensor).cuda()
              self.masks[i] = torch.from_numpy(augmentation_mask[k]).type(torch.cuda.FloatTensor).cuda()
              self.policy[i] = torch.from_numpy(augmentation_policy[k]).type(torch.cuda.FloatTensor).cuda()
              self.value[i] = torch.as_tensor(dat.value).type(torch.cuda.FloatTensor).cuda()


             #self.value = torch.from_numpy(values)
        ooo = 5
      

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx],self.masks[idx],self.policy[idx],self.value[idx]
        # board = self.boards[idx]
        # board = board.to('cuda')
        # board = board.type(torch.cuda.FloatTensor)

        # mask = self.masks[idx]
        # mask = board.to('cuda')
        # mask = board.type(torch.cuda.FloatTensor)
        
        # policy = self.policy[idx]
        # policy = policy.to('cuda')
        # policy = policy.type(torch.cuda.FloatTensor)

        # value = self.value[idx]
        # value = value.to('cuda')
        # value = value.type(torch.cuda.FloatTensor)

        # return board,mask, policy,value


class Config(collections.namedtuple(
    "Config", [
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "height",
        "width",
        "quiet",
        "observation_shape",
        "output_size",
        "path",
        "n_in_row"
    ])):
  """A config for the model/experiment."""
  pass


def _init_model_from_config(config):
  return model.UnetPolicyValueNet(config.width,config.height)


def watcher(fn):
  """A decorator to fn/processes that gives a logger and logs exceptions."""
  @functools.wraps(fn)
  def _watcher(*, config, num=None, **kwargs):
    """Wrap the decorated function."""
    name = fn.__name__
    if num is not None:
      name += "-" + str(num)
    with file_logger.FileLogger(config.path, name, config.quiet) as logger:
      print("{} started".format(name))
      logger.print("{} started".format(name))
      try:
        return fn(config=config, logger=logger, **kwargs)
      except Exception as e:
        logger.print("\n".join([
            "",
            " Exception caught ".center(60, "="),
            traceback.format_exc(),
            "=" * 60,
        ]))
        print("Exception caught in {}: {}".format(name, e))
        raise
      finally:
        logger.print("{} exiting".format(name))
        print("{} exiting".format(name))
  return _watcher


def _init_bot(config, game, evaluator_, evaluation):
  """Initializes a bot."""
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      evaluator_,
      solve=True,
      dirichlet_noise=noise,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=True,
      dont_return_chance_node=True)


def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.opt_print("Initial state:\n{}".format(state))
  while not state.is_terminal():
      root = bots[state.current_player()].mcts_search(state)

      policy = np.zeros(game.width*game.height)
      for c in root.children:
        policy[c.action] = c.explore_count
      policy = policy**(1 / temperature)
      policy /= policy.sum()
      best_action = root.best_child().action
      if len(actions) >= temperature_drop:
        action = best_action
      else:
        action = np.random.choice(len(policy), p=policy)
      ts = TrajectoryState(state.observation_tensor(), state.current_player(),
                          state.legal_actions_mask(),action, best_action, policy,
                          root.total_reward / root.explore_count)
      trajectory.states.append(ts)
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      logger.opt_print("Player {} sampled action: {}".format(
          state.current_player(), action_str))

      if True:
        logger.opt_print("\n" + str(state))
        logger.opt_print(ts)
        logger.opt_print("Finished {} sims".format(
            root.explore_count))
        logger.opt_print("Root:")
        logger.opt_print(root.to_str(state))
        logger.opt_print("Children:")
        logger.opt_print("\n{}".format(root.children_str(state)))
        logger.opt_print("optimal:" + str(len(actions) >= temperature_drop))
        if (len(actions) < temperature_drop):
          logger.opt_print("best action {}".format(state.action_to_string(state.current_player(), best_action)))
        logger.opt_print("---------------------")

      state.apply_action(action)
  logger.opt_print("Next state:\n{}".format(state))

  trajectory.returns = state.returns()
  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory


def update_checkpoint(logger, queue, model, az_evaluator):
  """Read the queue for a checkpoint to load, or an exit signal."""
  path = None
  while True:  # Get the last message, ignore intermediate ones.
    try:
      path = queue.get_nowait()
    except spawn.Empty:
      break
  if path:
    logger.print("Inference cache:", az_evaluator.cache_info())
    logger.print("Loading checkpoint", path)
    model.load_checkpoint(path)
    az_evaluator.clear_cache()
  elif path is not None:  # Empty string means stop this process.
    return False
  return True


@watcher
def actor(*, config, game, logger, queue):
  """An actor process runner that generates games and returns trajectories."""
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = AlphaZeroEvaluator(game, model)
  bots = [
      _init_bot(config, game, az_evaluator, False),
      _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return
    queue.put(_play_game(logger, game_num, game, bots, config.temperature,
                         config.temperature_drop))


@watcher
def evaluator(*, game, config, logger, queue):
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = Buffer(config.evaluation_window)
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = AlphaZeroEvaluator(game, model)
  random_evaluator = mcts.RandomRolloutEvaluator()

  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return

    az_player = game_num % 2
    difficulty = (game_num // 2) % config.eval_levels
    max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
    bots = [
        _init_bot(config, game, az_evaluator, True),
        mcts.MCTSBot(
            game,
            config.uct_c,
            max_simulations,
            random_evaluator,
            solve=True,
            verbose=True,
            dont_return_chance_node=True)
    ]
    if az_player == 1:
      bots = list(reversed(bots))

    trajectory = _play_game(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
    queue.put((difficulty, trajectory.returns[az_player]))

    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results.data)))


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger):
  """A learner that consumes the replay buffer and trains the network."""
  logger.also_to_stdout = True
  replay_buffer = Buffer(config.replay_buffer_size)
  learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
  logger.print("Initializing model")
  model = _init_model_from_config(config)
#   logger.print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
#                                            config.nn_depth))
#   logger.print("Model size:", model.num_trainable_variables, "variables")
  save_path = model.save_checkpoint(0)
  logger.print("Initial checkpoint:", save_path)
  broadcast_fn(save_path)

  data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

  stage_count = 7
  value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
  value_predictions = [stats.BasicStats() for _ in range(stage_count)]
  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
  evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
  total_trajectories = 0

  def trajectory_generator():
    """Merge all the actor queues into a single generator."""
    while True:
      found = 0
      for actor_process in actors:
        try:
          yield actor_process.queue.get_nowait()
        except spawn.Empty:
          pass
        else:
          found += 1
      if found == 0:
        time.sleep(0.01)  # 10ms

  def collect_trajectories():
    """Collects the trajectories from actors into the replay buffer."""
    num_trajectories = 0
    num_states = 0
    for trajectory in trajectory_generator():
      num_trajectories += 1
      num_states += len(trajectory.states)
      game_lengths.add(len(trajectory.states))
      game_lengths_hist.add(len(trajectory.states))

      p1_outcome = trajectory.returns[0]
      if p1_outcome > 0:
        outcomes.add(0)
      elif p1_outcome < 0:
        outcomes.add(1)
      else:
        outcomes.add(2)

      replay_buffer.extend(
          simple_model.TrainInput(
              s.observation, s.legals_mask, s.policy, p1_outcome)
          for s in trajectory.states)

      for stage in range(stage_count):
        # Scale for the length of the game
        index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
        n = trajectory.states[index]
        accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
        value_accuracies[stage].add(1 if accurate else 0)
        value_predictions[stage].add(abs(n.value))

      if num_states >= learn_rate:
        break
    return num_trajectories, num_states

  def learn(step):
    """Sample from the replay buffer, update weights and save a checkpoint."""
    losses = {
      "loss":0.0,
      "entropy":0.0,
      "n":0
    }
    replay_buffer.save('lastpickle')
    cgs = CustomGameDataset(replay_buffer)
    trainLoader = torch.utils.data.DataLoader(dataset = cgs, batch_size=64)
    model.policy_value_net.train()
    epochs = 200
    for k in range(epochs):
      for i, (inputs,masks, target_policy, target_value) in enumerate(trainLoader):
        #data = replay_buffer.sample(config.train_batch_size)

        loss = model.train_step(inputs,masks,target_policy,target_value,0.0005)
        losses["loss"]+=loss["loss"]
        losses["entropy"]+=loss["entripy"]
        losses["n"]+=1

    losses["loss"]/=losses["n"]
    losses["entropy"]/=losses["n"]


    # Always save a checkpoint, either for keeping or for loading the weights to
    # the actors. It only allows numbers, so use -1 as "latest".
    save_path = model.save_checkpoint(
        step if step % config.checkpoint_freq == 0 else -1)
   
    logger.print(losses)
    logger.print("Checkpoint saved:", save_path)
    return save_path, losses

  last_time = time.time() - 60
  for step in itertools.count(1):
    for value_accuracy in value_accuracies:
      value_accuracy.reset()
    for value_prediction in value_predictions:
      value_prediction.reset()
    game_lengths.reset()
    game_lengths_hist.reset()
    outcomes.reset()

    num_trajectories, num_states = collect_trajectories()
    total_trajectories += num_trajectories
    now = time.time()
    seconds = now - last_time
    last_time = now

    logger.print("Step:", step)
    logger.print(
        ("Collected {:5} states from {:3} games, {:.1f} states/s. "
         "{:.1f} states/(s*actor), game length: {:.1f}").format(
             num_states, num_trajectories, num_states / seconds,
             num_states / (config.actors * seconds),
             num_states / num_trajectories))
    logger.print("Buffer size: {}. States seen: {}".format(
        len(replay_buffer), replay_buffer.total_seen))

    save_path, losses = learn(step)

    for eval_process in evaluators:
      while True:
        try:
          difficulty, outcome = eval_process.queue.get_nowait()
          evals[difficulty].append(outcome)
        except spawn.Empty:
          break

    batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
    batch_size_stats.add(1)
    data_log.write({
        "step": step,
        "total_states": replay_buffer.total_seen,
        "states_per_s": num_states / seconds,
        "states_per_s_actor": num_states / (config.actors * seconds),
        "total_trajectories": total_trajectories,
        "trajectories_per_s": num_trajectories / seconds,
        "queue_size": 0,  # Only available in C++.
        "game_length": game_lengths.as_dict,
        "game_length_hist": game_lengths_hist.data,
        "outcomes": outcomes.data,
        "value_accuracy": [v.as_dict for v in value_accuracies],
        "value_prediction": [v.as_dict for v in value_predictions],
        "eval": {
            "count": evals[0].total_seen,
            "results": [sum(e.data) / len(e) if e else 0 for e in evals],
        },
        "batch_size": batch_size_stats.as_dict,
        "batch_size_hist": [0, 1],
        "loss": {
            "policy": losses,
        },
        "cache": {  # Null stats because it's hard to report between processes.
            "size": 0,
            "max_size": 0,
            "usage": 0,
            "requests": 0,
            "requests_per_s": 0,
            "hits": 0,
            "misses": 0,
            "misses_per_s": 0,
            "hit_rate": 0,
        },
    })
    logger.print()

    if config.max_steps > 0 and step >= config.max_steps:
      break

    broadcast_fn(save_path)


def alpha_zero(config: Config):
  """Start all the worker processes for a full alphazero setup."""
  width = config.width
  height = config.height
  game = main.MNKGame(width=width,height=height,n_in_row=config.n_in_row )
  config = config._replace(
      observation_shape=[width,height],
      output_size=width*height)

  print("Starting game",game)
  
  path = 'a0'
  config = config._replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit("{} isn't a directory".format(path))
  print("Writing logs and checkpoints to:", path)

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

  actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                         "num": i})
            for i in range(config.actors)]
  evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                 "num": i})
                for i in range(config.evaluators)]

  def broadcast(msg):
    for proc in actors + evaluators:
      proc.queue.put(msg)

  try:
    learner(game=game, config=config, actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators, broadcast_fn=broadcast)
  except (KeyboardInterrupt, EOFError):
    print("Caught a KeyboardInterrupt, stopping early.")
  finally:
    broadcast("")
    # for actor processes to join we have to make sure that their q_in is empty,
    # including backed up items
    for proc in actors:
      while proc.exitcode is None:
        while not proc.queue.empty():
          proc.queue.get_nowait()
        proc.join(JOIN_WAIT_DELAY)
    for proc in evaluators:
      proc.join()
def start():
    print(444)
    config = Config(learning_rate = 0.002,
        uct_c=1.5,
        max_simulations=500,
        train_batch_size = 2**6,
        replay_buffer_size = 2**12,
        replay_buffer_reuse = 16,
        weight_decay = 0.0001,
        policy_epsilon = 0.25,
        policy_alpha = 1,
        temperature = 1,
        temperature_drop=6,
        checkpoint_freq = 10,
        actors=1,
        evaluators=0,
        evaluation_window=100,
        eval_levels=7,
        max_steps=0,
        quiet=False,
        width=8,
        height=8,
        observation_shape=[8,8],
        output_size=8*8,
        path=".",
        n_in_row=5
       )
    alpha_zero(config)
    
if __name__ == '__main__':
    print(4444)
    start()