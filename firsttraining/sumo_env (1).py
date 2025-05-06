import xml.etree.ElementTree as ET
import gym, numpy as np, traci
from gym import spaces
from sumolib import checkBinary, net as sumo_net

class SumoEnv(gym.Env):
    """Gym wrapper for your custom SUMO network."""
    metadata = {'render.modes': ['human']}

    def __init__(self, sumo_cfg: str, max_steps: int = 200):
        super().__init__()
        self.sumo_binary = checkBinary("sumo")
        self.sumo_cfg    = sumo_cfg
        self.max_steps   = max_steps
        self.step_count  = 0

        # ─── parse sim.sumocfg to find the net file ─────────────────────────
        root = ET.parse(sumo_cfg).getroot()
        net_file = root.find("input/net-file").attrib["value"]
        # ─── load the network and list all non-internal edges ─────────────
        net_obj = sumo_net.readNet(net_file)
        self.incoming_edges = [
            e.getID() for e in net_obj.getEdges()
            if not e.getID().startswith(":")
        ]

        # ─── define spaces ─────────────────────────────────────────────────
        # waiting time on each edge, unbounded ≥0
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(len(self.incoming_edges),),
            dtype=np.float32
        )
        # assume your TLS has 4 phases; adjust if needed
        self.action_space = spaces.Discrete(4)

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true"
        ])
        self.step_count = 0
        # warm up one step so vehicles at depart=0 appear
        traci.simulationStep()
        return self._get_obs()

    def step(self, action: int):
        # apply a light phase
        tls_id = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tls_id, action)

        traci.simulationStep()
        self.step_count += 1

        obs    = self._get_obs()
        reward = - float(obs.sum())      # minimize total waiting
        done   = (self.step_count >= self.max_steps)
        info   = {}
        return obs, reward, done, info

    def _get_obs(self):
        # waitingTime on each edge
        waits = [
            traci.edge.getWaitingTime(edgeID)
            for edgeID in self.incoming_edges
        ]
        return np.array(waits, dtype=np.float32)

    def close(self):
        if traci.isLoaded():
            traci.close()