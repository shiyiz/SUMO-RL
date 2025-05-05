# test_sumo_env.py
from gym.envs.registration import register
import gym

# Register your environment
register(
    id="SmokeSumo-v0",
    entry_point="sumo_env:SumoEnv",
    max_episode_steps=50,
)

# Instantiate & test
env = gym.make("SmokeSumo-v0", sumo_cfg="oneIntersection.sumocfg")
obs = env.reset()
print("Initial observation:", obs)

for i in range(5):
    obs, reward, done, info = env.step(0)
    print(f" Step {i+1}  obs={obs}  done={done}")
    if done:
        break

env.close()
