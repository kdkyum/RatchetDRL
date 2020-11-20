import argparse

import torch

import ppo.net as net
from ppo.ppo import ppo
from utils.run_utils import ExperimentGrid

nn_names = sorted(
    name
    for name in net.__dict__
    if name.islower() and not name.startswith("__") and callable(net.__dict__[name])
)

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default="ppo-delay")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="rnn",
    help="NN architecture: " + " | ".join(nn_names) + " (default: mlp)",
)
parser.add_argument("--N", metavar="N", type=int, default=1)
parser.add_argument("--delay", type=int, default=50)
parser.add_argument("--env", type=str, default='A')
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    train_iter = 625
    M = int(2 ** 10 / args.N)
    B = 4096 if args.N < 8 else 32 * M
    if args.N >= 128:
        M = 8
        B = 256

    eg = ExperimentGrid(name=args.name)
    eg.add("seed", args.seed)
    eg.add("epochs", 400)
    eg.add("lam", 0.95)
    eg.add("gamma", 0.999)
    eg.add("num_trjs", M)
    eg.add("batch_size", B)
    eg.add("test_batch_size", 2 * B)
    eg.add("train_pi_iters", train_iter)
    eg.add("train_v_iters", train_iter)
    if args.env=='A':
        eg.add("env_name", "CFR_A_delay")
    else:
        eg.add("env_name", "CFR_B_delay")
    eg.add("arch", args.arch, "Arch-", True)
    eg.add("env_kwargs:N", args.N, "N", True)
    eg.add("env_kwargs:a", 1)
    eg.add("env_kwargs:dt", 0.001)
    eg.add("env_kwargs:U", 5)
    eg.add("env_kwargs:delay", args.delay, "delay", True)
    eg.run(ppo)
