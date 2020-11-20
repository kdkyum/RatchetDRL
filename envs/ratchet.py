import numpy as np
from envs.base_env import Env, Discrete, Circle


__all__ = ["CFR_A", "CFR_B", "CFR_A_delay", "CFR_B_delay", "CFR_open"]


class CollectiveFlashingRatchet(Env):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1):
        # M: number of ensembles
        # N: number of particles
        # a: system size
        # T: temperature of a heat bath
        # U: potential
        # e: friction coefficient
        # dt: time step size
        self.observation_space = Circle(N)
        self.action_space = Discrete(2)
        self.M = M
        self.N = N
        self.T = T
        self.U = U
        self.e = e
        self.a = a
        self.dt = dt
        self.reset(self.M)

    def potential(self, x):
        raise NotImplementedError

    def force(self, x):
        raise NotImplementedError

    def get_obs(self):
        obs = np.stack(
            [
                np.cos(2 * np.pi * self.x / self.a),
                np.sin(2 * np.pi * self.x / self.a),
            ],
            axis=-1,
        )
        return obs

    def get_greedy_action(self):
        def get_action(obs):
            z = obs[:, :, 0] + 1j * obs[:, :, 1]
            x = np.angle(z) * self.a / (2 * np.pi)
            ret = np.sum(self.force(x), axis=-1)
            return (ret > 0).astype(np.float32)

        return get_action

    def step(self, action):
        _rand = np.sqrt(2 * self.T * self.e * self.dt) * np.random.randn(self.M, self.N)
        _drift = action.reshape(-1, 1) * self.force(self.x) * self.dt
        self.x += (_drift + _rand) / self.e
        self.t += self.dt
        obs = self.get_obs()
        rew = (_drift + _rand).mean(axis=-1) / self.dt
        rew = rew.astype(np.float32)
        return obs, rew, False, None

    def reset(self, M=None):
        if M is None:
            self.x = self.observation_space.sample(self.M)
        else:
            self.M = M
            self.x = self.observation_space.sample(self.M)
        self.t = 0
        return self.get_obs()

    def seed(self, seed=None):
        np.random.seed(seed)
        self.seed = seed
        return seed


class CFR_A(CollectiveFlashingRatchet):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1):
        super().__init__(M, N, a, T, U, e, dt, bins)

    def potential(self, x):
        U = self.U * (
            np.sin(2 * np.pi * x / self.a) + 0.25 * np.sin(4 * np.pi * x / self.a)
        )
        return U

    def force(self, x):
        F = (
            -self.U
            * np.pi
            * (2 * np.cos(2 * np.pi * x / self.a) + np.cos(4 * np.pi * x / self.a))
            / self.a
        )
        return F


class CFR_B(CollectiveFlashingRatchet):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1):
        super().__init__(M, N, a, T, U, e, dt, bins)

    def potential(self, x):
        m = np.mod(x, self.a)
        ret = 3 * self.U / self.a * m
        idx = (m >= self.a / 3) & (m <= self.a)
        ret[idx] = self.U - self.U * (m[idx] - self.a / 3) / (self.a - self.a / 3)
        return ret

    def force(self, x):
        m = np.mod(x, self.a)
        ret = -np.ones_like(x) * self.U * 3 / self.a
        idx = (m >= self.a / 3) & (m <= self.a)
        ret[idx] = self.U / (self.a - self.a / 3)
        return ret


class CFR_A_delay(CFR_A):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1, delay=20):
        self.delay = delay
        self.delayed_queue = []
        super().__init__(M, N, a, T, U, e, dt, bins)
        self.x_max1 = self.a * (np.arctan(np.sqrt(-3 + 2 * np.sqrt(3))) / np.pi)
        self.x_max2 = self.x_max1 + self.a
        self.x_min = -0.19035916268766679 + self.a

    def _displacement(self, x, x0):
        tmp = x[x < self.x_max1] + 1
        x[x < self.x_max1] = tmp
        disp = x - self.x_min - x0
        return disp

    def get_mnd_action(self, x0=0):
        # Maximize the net displacement (MND)
        def get_action(obs):
            z = obs[:, :, 0] + 1j * obs[:, :, 1]
            x = np.angle(z) * self.a / (2 * np.pi)
            ret = -np.sum(self._displacement(x, x0), axis=-1)
            return (ret > 0).astype(np.float32)

        return get_action

    def step(self, action):
        self.delayed_queue.append(action)
        if len(self.delayed_queue) > 0:
            _act = self.delayed_queue.pop(0)
            _act = _act.reshape(-1, 1)
        _rand = np.sqrt(2 * self.T * self.e * self.dt) * np.random.randn(self.M, self.N)
        _drift = _act * self.force(self.x) * self.dt
        self.x += (_drift + _rand) / self.e
        obs = self.get_obs()
        rew = (_drift + _rand).mean(axis=-1) / self.dt
        rew = rew.astype(np.float32)
        return obs, rew, False, None

    def get_act_buffer(self):
        return self.delayed_queue

    def reset(self, M=None):
        if M is None:
            self.x = self.observation_space.sample(self.M)
        else:
            self.M = M
            self.x = self.observation_space.sample(self.M)

        for i in range(self.delay):
            if len(self.delayed_queue) < self.delay:
                a = self.action_space.sample(self.M)
                self.delayed_queue.append(a)

        return self.get_obs()


class CFR_B_delay(CFR_A_delay):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1, delay=20):
        super().__init__(M, N, a, T, U, e, dt, bins, delay)

    def potential(self, x):
        m = np.mod(x, self.a)
        ret = 3 * self.U / self.a * m
        idx = (m >= self.a / 3) & (m <= self.a)
        ret[idx] = self.U - self.U * (m[idx] - self.a / 3) / (self.a - self.a / 3)
        return ret

    def force(self, x):
        m = np.mod(x, self.a)
        ret = -np.ones_like(x) * self.U * 3 / self.a
        idx = (m >= self.a / 3) & (m <= self.a)
        ret[idx] = self.U / (self.a - self.a / 3)
        return ret


class CFR_open(CollectiveFlashingRatchet):
    def __init__(self, M=1, N=1, a=1, T=1, U=5, e=1, dt=0.001, bins=-1):
        super().__init__(M, N, a, T, U, e, dt, bins)
        self.observation_space.shape = (self.N, 1)

    def get_obs(self):
        t = np.ones_like(self.x) * self.t
        return t.reshape(self.M, self.N, 1)

    def get_periodic_action(self, t, t1=0.04, t2=0.03):
        T = t1 + t2
        t = np.mod(t, T)
        return (t > t1).astype(np.float32)

    def potential(self, x):
        U = self.U * (
            np.sin(2 * np.pi * x / self.a) + 0.25 * np.sin(4 * np.pi * x / self.a)
        )
        return U

    def force(self, x):
        F = (
            -self.U
            * np.pi
            * (2 * np.cos(2 * np.pi * x / self.a) + np.cos(4 * np.pi * x / self.a))
            / self.a
        )
        return F
