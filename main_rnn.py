import argparse

from ppo.ppo_delay import ppo
from utils.run_utils import ExperimentGrid

env_names = ["A_delay", "B_delay"]

parser = argparse.ArgumentParser(
    description="Running time-delayed PPO algorithm for a collective flashing ratchet model."
)
parser.add_argument(
    "-n", "--name", type=str, default="ppo-delay", help="Experiment name"
)
parser.add_argument(
    "--env",
    metavar="ENV",
    default="A_delay",
    choices=env_names,
    help="Environments: " + " | ".join(env_names) + " (default: A_delay)",
)
parser.add_argument("--N", metavar="N", type=int, default=1, help="Number of particles")
parser.add_argument(
    "--delay", type=int, default=10, help="Number of delayed time step (default: 10)"
)
parser.add_argument(
    "--gamma",
    metavar="GAM",
    type=float,
    default=0.999,
    help="Discounting factor (default: 0.999)",
)
parser.add_argument(
    "--lam",
    metavar="LAM",
    type=float,
    default=0.95,
    help="Parameter of Generalized Advantage Estimator (GAE) (default: 0.95)",
)
parser.add_argument(
    "--epoch", type=int, default=400, help="Number of training epochs (default: 400)"
)
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
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
    eg.add("epochs", args.epoch)
    eg.add("lam", args.lam)
    eg.add("gamma", args.gamma)
    eg.add("num_trjs", M)
    eg.add("batch_size", B)
    eg.add("test_batch_size", 2 * B)
    eg.add("train_pi_iters", train_iter)
    eg.add("train_v_iters", train_iter)
    eg.add("env_name", "CFR_%s" % args.env)
    eg.add("arch", "rnn", "Arch-", True)
    eg.add("env_kwargs:N", args.N, "N", True)
    eg.add("env_kwargs:U", 5)
    eg.add("env_kwargs:delay", args.delay, "delay", True)
    eg.run(ppo)
