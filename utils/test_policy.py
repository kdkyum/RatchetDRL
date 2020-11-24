import os
import os.path as osp

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from misc.logger import EpochLogger


def load_policy_and_env(fpath, itr="last", deterministic=False):
    # handle which epoch to load from
    if itr == "last":
        # check filenames for epoch (AKA iteration) numbers, find maximum value
        pytsave_path = osp.join(fpath, "pyt_save")
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [
            int(x.split(".")[0][5:])
            for x in os.listdir(pytsave_path)
            if len(x) > 8 and "model" in x
        ]
        itr = "%d" % max(saves) if len(saves) > 0 else ""
    else:
        assert isinstance(
            itr, int
        ), "Bad value provided for itr (needs to be int or 'last')."
        itr = "%d" % itr
    get_action, get_p = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, "vars" + itr + ".pkl"))
        env = state["env"]
    except:
        env = None

    return env, get_action, get_p


def load_pytorch_policy(fpath, itr, deterministic=False, device="cuda"):
    fname = osp.join(fpath, "pyt_save", "model" + itr + ".pt")
    print("Loading from %s.\n" % fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        if deterministic:
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
                logit = model.pi.logits_net(x)
                _, indices = torch.max(logit, 1)
                action = indices.float().cpu().numpy()
            return action
        else:
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
                action = model.act(x)
            return action

    def get_p(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            logit = model.pi.logits_net(x)
        return F.softmax(logit, dim=-1).cpu().numpy()

    return get_action, get_p


def run_policy(env, get_action, log=False, trj_len=50000):
    assert env is not None, (
        "Environment not found!\n\n It looks like the environment wasn't saved, "
        + "and we can't run the agent in it. :( \n\n Check out the readthedocs "
        + "page on Experiment Outputs for how to handle this situation."
    )
    o, ep_ret, ep_len = env.get_obs(), 0, 0
    if log:
        logger = EpochLogger()
        obs_buf, rew_buf, act_buf = [], [], []

    for i in range(trj_len):
        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_len += 1
        ep_ret += r
        if log:
            obs_buf.append(o)
            rew_buf.append(r)
            act_buf.append(a)

    if log:
        obs_buf = np.stack(obs_buf, axis=1)
        rew_buf = np.stack(rew_buf, axis=1)
        act_buf = np.stack(act_buf, axis=1)
        logger.store(Ret=ep_ret / ep_len, EpLen=ep_len)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.dump_tabular()
        return obs_buf, act_buf, rew_buf
    else:
        return _, _, ep_ret / ep_len


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str)
    parser.add_argument("--len", "-l", type=int, default=50000)
    parser.add_argument("--itr", "-i", type=int, default=-1)
    parser.add_argument("--deterministic", "-d", action="store_true")
    args = parser.parse_args()

    s = "deterministic" if args.deterministic else "stochastic"
    print("Testing policy (%s)...\n" % s + "=" * 50 + "\n")
    env, get_action, get_p = load_policy_and_env(
        args.fpath, args.itr if args.itr >= 0 else "last", args.deterministic
    )
    _, _, ret = run_policy(env, get_action, trj_len=args.len)
    mean_r, std_r, max_r, min_r = ret.mean(), ret.std(), ret.max(), ret.min()
    df = pd.DataFrame(
        [{"AvgRet": mean_r, "StdRet": std_r, "MaxRet": max_r, "MinRet": min_r}]
    )
    if args.deterministic:
        df.to_csv(osp.join(args.fpath, "test_deterministic.csv"), index=False)
    else:
        df.to_csv(osp.join(args.fpath, "test_stochastic.csv"), index=False)

    print("=" * 50)
