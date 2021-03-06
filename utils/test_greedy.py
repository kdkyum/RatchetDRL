import os

import numpy as np
import pandas as pd
from itertools import product

from copy import deepcopy

import envs


def run_policy(env, get_action, log=False, burn_in=5000, trj_len=4000):
    num_trj = env.M
    o, ep_ret, ep_len = env.get_obs(), 0, 0

    for i in range(burn_in):
        a = get_action(o)
        o, r, _, _ = env.step(a)

    for i in range(trj_len):
        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_len += 1
        ep_ret += r

    return ep_ret / ep_len


def run_greedy(N, delay, env_name, trj_len, **env_kwargs):
    burn_in = 2000
    env_kwargs["N"] = N
    env_kwargs["delay"] = delay
    env = envs.__dict__[env_name](**env_kwargs)
    get_action = env.get_greedy_action()
    rew = run_policy(env, get_action, burn_in=burn_in, trj_len=trj_len)
    env_kwargs["trj_len"] = trj_len
    env_kwargs["env_name"] = env_name
    env_kwargs["rew_mean"] = rew.mean()
    env_kwargs["rew_std"] = rew.std()
    ret = deepcopy(env_kwargs)
    print(ret)
    return ret


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CFR_A_delay")
    parser.add_argument("--num_cpu", type=int, default=8)
    parser.add_argument("--num_trj", type=int, default=100)
    parser.add_argument("--trj_len", type=int, default=50000)
    parser.add_argument("--save", type=str, default=".")
    args = parser.parse_args()

    Ns = [2 ** i for i in range(7)]
    delays = np.arange(0, 55, 5)
    configs = [Ns, delays]
    star = product(*configs)

    env_kwargs = {}
    env_kwargs["M"] = args.num_trj
    run = partial(run_greedy, env_name=args.env, trj_len=args.trj_len, **env_kwargs)
    with Pool(args.num_cpu) as p:
        rets = list(p.starmap(run, star))

    df = pd.DataFrame(rets)
    df.to_csv(os.path.join(args.save, "greedy_delay.csv"), index=False)
