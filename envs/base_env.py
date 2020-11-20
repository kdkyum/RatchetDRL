import numpy as np


class Env(object):
    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def seed(self, seed=None):
        return

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)


class Space(object):
    def sample(self):
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, n):
        assert n > 0
        self.n = n

    def sample(self, M=1):
        return np.random.randint(self.n, size=(M,))

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n


class Circle(Space):
    def __init__(self, N):
        assert N > 0
        self.N = N
        self.shape = (N, 2)

    def sample(self, M=1):
        return np.random.uniform(0, 2 * np.pi, size=(M, self.N))

    def __repr__(self):
        return "%d: Circle" % self.N
