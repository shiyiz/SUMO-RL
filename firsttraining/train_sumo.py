# train_sumo.py
from gym.envs.registration import register
register(
    id="CustomSumo-v0",
    entry_point="sumo_env:SumoEnv",
    max_episode_steps=200,
)

import gym
from stable_baselines3 import PPO

# create the env
env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=200)

# set up and train
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
model.learn(total_timesteps=50000)
model.save("ppo_sumo")
env.close()
