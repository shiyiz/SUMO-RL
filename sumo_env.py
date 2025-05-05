# sumo_env.py
import gym, numpy as np, traci
from sumolib import checkBinary
from gym import spaces

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, max_steps=50):
        super().__init__()
        self.sumo_binary = checkBinary("sumo")
        self.sumo_cfg    = sumo_cfg
        self.max_steps   = max_steps
        self.step_count  = 0

        # 1) List the edges you want to observe
        self.incoming_edges = ['A0A1', 'A0B0', 'A1A0', 'A1B1', 'B0A0', 'B0B1', 'B1A1', 'B1B0']  # ← replace with your real IDs

        # 2) Define observation_space to match
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(len(self.incoming_edges),),
            dtype=np.float32
        )

        # You'll still pick from e.g. 2 phases
        self.action_space = spaces.Discrete(2)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([self.sumo_binary, "-c", self.sumo_cfg, "--no-step-log", "true"])
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        # apply your action here, e.g. traci.trafficlight.setPhase(...)
        traci.simulationStep()
        self.step_count += 1

        obs    = self._get_obs()
        reward = self._compute_reward()   # you’ll define this
        done   = (self.step_count >= self.max_steps)
        info   = {}
        return obs, reward, done, info

    def _get_obs(self):
        waits = [traci.edge.getWaitingTime(eid) for eid in self.incoming_edges]
        return np.array(waits, dtype=np.float32)

    def _compute_reward(self):
        # e.g. negative sum of waits
        return - float(sum(
            traci.edge.getWaitingTime(edge_id)
            for edge_id in self.incoming_edges
        ))

    def close(self):
        if traci.isLoaded():
            traci.close()
