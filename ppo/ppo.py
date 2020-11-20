import time

import numpy as np
import torch
from torch.optim import Adam

import envs
import ppo.core as core
from misc.logger import EpochLogger, setup_logger_kwargs
from misc.sampler import BatchSampler

env_names = sorted(
    name
    for name in envs.__dict__
    if name.islower() and not name.startswith("__") and callable(envs.__dict__[name])
)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, num_trjs, trj_len, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((num_trjs, trj_len, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.adv_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.rew_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.ret_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.val_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.logp_buf = np.zeros((num_trjs, trj_len), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.trj_len = 0, 0, trj_len

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.trj_len + 1  # buffer has to have room so you can store
        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
        self.logp_buf[:, self.ptr] = logp
        self.ptr += 1

    def get(self, device="cpu"):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.trj_len  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
            ret=self.ret_buf,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in data.items()
        }

    def finish_path(self, v):
        v = v.reshape(-1, 1)
        rews = np.append(self.rew_buf, v, axis=1)
        vals = np.append(self.val_buf, v, axis=1)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:, :-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
        self.adv_buf = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = core.discount_cumsum(rews, self.gamma)[:, :-1]

        self.path_start_idx = self.ptr


def ppo(
    env_name,
    env_kwargs=dict(),
    actor_critic=core.ActorCritic,
    arch="mlp",
    ac_kwargs=dict(),
    seed=0,
    num_trjs=100,
    trj_len=2000,
    epochs=50,
    batch_size=4000,
    test_batch_size=10000,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=1000,
    train_v_iters=1000,
    lam=0.96,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    burn_in=2000,
    device="cuda:0",
):
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs["M"] = num_trjs
    env = envs.__dict__[env_name](**env_kwargs)

    # Instantiate environment
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, arch, **ac_kwargs)
    ac.to(device)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log("NN architecture: %s" % arch)
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, num_trjs, trj_len, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = (
            torch.as_tensor(clipped, dtype=torch.float32, device=device).mean().item()
        )
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(obs, ret):
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    train_sampler = BatchSampler(num_trjs, trj_len, batch_size, device=device)
    test_sampler = BatchSampler(
        num_trjs, trj_len, test_batch_size, device=device, train=False
    )

    def validate_pi(obs, act, adv, logp_old):
        loss_pi, approx_kl, ent, clipfrac = 0, 0, 0, 0
        with torch.no_grad():
            for batch in test_sampler:
                pi, logp = ac.pi(obs[batch], act[batch])
                ratio = torch.exp(logp - logp_old[batch])
                clip_adv = (
                    torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[batch]
                )
                loss_pi += -(torch.min(ratio * adv[batch], clip_adv)).sum()

                # Useful extra info
                approx_kl += (logp_old[batch] - logp).sum().item()
                ent += pi.entropy().sum().item()
                clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                clipfrac += (
                    torch.as_tensor(clipped, dtype=torch.float32, device=device)
                    .sum()
                    .item()
                )

        loss_pi = loss_pi / test_sampler.size
        pi_info = dict(
            kl=approx_kl / test_sampler.size,
            ent=ent / test_sampler.size,
            cf=clipfrac / test_sampler.size,
        )
        return loss_pi, pi_info

    def validate_v(obs, ret):
        loss_v = 0
        with torch.no_grad():
            for batch in test_sampler:
                loss_v += ((ac.v(obs[batch]) - ret[batch]) ** 2).sum()
        loss_v = loss_v / test_sampler.size
        return loss_v

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get(device)

        obs, act, adv, logp_old, ret = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logp"],
            data["ret"],
        )
        pi_l_old, pi_info_old = validate_pi(obs, act, adv, logp_old)
        pi_l_old = pi_l_old.item()
        v_l_old = validate_v(obs, ret).item()

        # Train policy with multiple steps of sgd
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            batch = next(train_sampler)

            loss_pi, pi_info = compute_loss_pi(
                obs[batch], act[batch], adv[batch], logp_old[batch]
            )
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            batch = next(train_sampler)
            loss_v = compute_loss_v(obs[batch], ret[batch])
            loss_v.backward()
            vf_optimizer.step()

        loss_pi, pi_info = validate_pi(obs, act, adv, logp_old)
        loss_pi = loss_pi.item()
        loss_v = validate_v(obs, ret).item()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi - pi_l_old),
            DeltaLossV=(loss_v - v_l_old),
        )

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for _ in range(burn_in):
        a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
        o, r, d, _ = env.step(a)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(trj_len):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r * env.dt, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o
            epoch_ended = t == trj_len - 1

            if epoch_ended:
                a, v, logp = ac.step(
                    torch.as_tensor(o, dtype=torch.float32, device=device)
                )
                buf.finish_path(v)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)
        logger.store(Ret=ep_ret / ep_len, EpLen=ep_len)

        # Perform PPO update!
        update()
        ep_ret, ep_len = 0, 0

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("Ret", with_min_and_max=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * trj_len)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--trj_len", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="ppo")
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(
        "CFR_A",
        actor_critic=core.ActorCritic,
        gamma=args.gamma,
        seed=args.seed,
        trj_len=args.trj_len,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
