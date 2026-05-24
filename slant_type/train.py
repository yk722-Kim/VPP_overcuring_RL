import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from slant_env import SlantEnv, N_LAYERS


def make_env():
    return SlantEnv()


def evaluate_deterministic(model):
    env = SlantEnv()
    obs, _ = env.reset()
    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO for the slant-type overhang grayscale schedule."
    )
    parser.add_argument("--total_timesteps", type=int, default=400_000)
    parser.add_argument("--save_dir", type=str, default="./ppo_slant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env for _ in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(args.save_dir, "tb"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=max(1000, args.total_timesteps // 50),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(os.path.join(args.save_dir, "final_model"))

    info = evaluate_deterministic(model)

    print()
    print("=" * 60)
    print("Final deterministic policy")
    print("=" * 60)
    print(f"g_history                       = {info['g_history']}")
    print(f"t_history                       = {[f'{t:.4f}' for t in info['t_history']]}")
    print(f"total_z_gel_um                  = {info['total_z_gel_um']:.3f}")
    print(f"total_z_overcure_um             = {info['total_z_overcure_um']:.3f}")
    print(f"total_objective_um              = {info['total_objective_um']:.3f}")
    print(f"total_exposure_time_s           = {info['total_exposure_time_s']:.3f}")
    print()
    print("Per-overhang results")
    print("-" * 60)

    for idx in range(N_LAYERS):
        print(
            f"{idx + 1:02d}: "
            f"z_gel={info['per_overhang_z_gel_um'][idx]:.3f} um, "
            f"z_overcure={info['per_overhang_z_overcure_um'][idx]:.3f} um, "
            f"z_gel_only_band={info['per_overhang_z_gel_only_band_um'][idx]:.3f} um"
        )


if __name__ == "__main__":
    main()
