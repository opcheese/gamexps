
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import mcts
import collections
import torch

class CacheInfo(collections.namedtuple("CacheInfo", [
    "hits", "misses", "size", "max_size"])):
  """Info for LRUCache."""

  @property
  def usage(self):
    return self.size / self.max_size if self.max_size else 0

  @property
  def total(self):
    return self.hits + self.misses

  @property
  def hit_rate(self):
    return self.hits / self.total if self.total else 0


class LRUCache(object):
  """A Least Recently Used cache.
  This is more general than functools.lru_cache since that one requires the
  key to also be the input to the function to generate the value, which
  isn't possible when the input is not hashable, eg a numpy.ndarray.
  """

  def __init__(self, max_size):
    self._max_size = max_size
    self._data = collections.OrderedDict()
    self._hits = 0
    self._misses = 0

  def clear(self):
    self._data.clear()
    self._hits = 0
    self._misses = 0

  def make(self, key, fn):
    """Return the value, either from cache, or make it and save it."""
    try:
      val = self._data.pop(key)  # Take it out.
      self._hits += 1
    except KeyError:
      self._misses += 1
      val = fn()
      if len(self._data) >= self._max_size:
        self._data.popitem(False)
    self._data[key] = val  # Insert/reinsert it at the back.
    return val

  def get(self, key):
    """Get the value and move it to the back, or return None on a miss."""
    try:
      val = self._data.pop(key)  # Take it out.
      self._data[key] = val  # Reinsert it at the back.
      self._hits += 1
      return val
    except KeyError:
      self._misses += 1
      return None

  def set(self, key, val):
    """Set the value."""
    self._data.pop(key, None)  # Take it out if it existed.
    self._data[key] = val  # Insert/reinsert it at the back.
    if len(self._data) > self._max_size:
      self._data.popitem(False)
    return val

  def info(self):
    return CacheInfo(self._hits, self._misses, len(self._data), self._max_size)

  def __len__(self):
    return len(self._data)

class AlphaZeroEvaluator(mcts.Evaluator):
  """An AlphaZero MCTS Evaluator."""

  def __init__(self, game, model, cache_size=2**13):
    """An AlphaZero MCTS Evaluator."""

    self._model = model
    self._cache = LRUCache(cache_size)

  def cache_info(self):
    return self._cache.info()

  def clear_cache(self):
    self._cache.clear()

  def _inference(self, state):
    # Make a singleton batch
    tens = state.observation_tensor()
    tens = np.expand_dims(tens, axis=0)
    t = torch.from_numpy(tens)
    t = t.to('cuda')
    t = t.type(torch.cuda.FloatTensor)

    msk = state.legal_actions_mask()
    m = torch.from_numpy(msk)
    m = m.to('cuda')
    m = m.type(torch.cuda.FloatTensor)
    # ndarray isn't hashable
    cache_key = str(state)
    
    policy, value = self._cache.make(
        cache_key, lambda: self._model.inference(t, m))

    return value[0, 0], policy[0]  # Unpack batch

  def evaluate(self, state):
    """Returns a value for the given state."""
    value, _ = self._inference(state)
    value = value.detach().cpu()
    return np.array([value, -value])

  def prior(self, state):
      # Returns the probabilities for all actions.
    _, policy = self._inference(state)
    #print(888)
    #print(policy)
    return [(action, policy[action]) for action in state.legal_actions()]