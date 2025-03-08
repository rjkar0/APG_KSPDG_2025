import numpy as np
from kspdg import LBG1_LG3_I2_V1
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner

import datetime

# Class to define the agent, here being the Game Theoretic controller
class NaivePursuitAgent(KSPDGBaseAgent):
    """An agent that naively burns directly toward it's target"""
    def __init__(self):
        super().__init__()


    def get_action(self, observation):
        """ compute agent's action given observation

        This function is necessary to define as it overrides 
        an abstract method
        """

        obs = observation

        # Assign variables to individual values in state vector
        t, mb, mbf, b_px, b_py, b_pz, b_vx, b_vy, b_vz, l_px, l_py, l_pz, l_vx, l_vy, l_vz, g_px, g_py, g_pz, g_vx, g_vy, g_vz = range(21)

        # Calculate relative position and velocity

        # Bandit relative to guard
        b_px_rg = obs[g_px] - obs[b_px]
        b_py_rg = obs[g_py] - obs[b_py]
        b_pz_rg = obs[g_pz] - obs[b_pz]
        b_vx_rg = obs[g_vx] - obs[b_vx]
        b_vy_rg = obs[g_vy] - obs[b_vy]
        b_vz_rg = obs[g_vz] - obs[b_vz]

        b_p_rg = (b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        b_v_rg = (b_vx_rg**2 + b_vy_rg**2 + b_vz_rg**2)**0.5

        # Bandit relative to lady
        b_px_rl = obs[l_px] - obs[b_px]
        b_py_rl = obs[l_py] - obs[b_py]
        b_pz_rl = obs[l_pz] - obs[b_pz]
        b_vx_rl = obs[l_vx] - obs[b_vx]
        b_vy_rl = obs[l_vy] - obs[b_vy]
        b_vz_rl = obs[l_vz] - obs[b_vz]

        b_p_rl = (b_px_rl**2 + b_py_rl**2 + b_pz_rl**2)**0.5
        b_v_rl = (b_vx_rl**2 + b_vy_rl**2 + b_vz_rl**2)**0.5

        # Guard relative to lady
        g_px_rl = obs[l_px] - obs[g_px]
        g_py_rl = obs[l_py] - obs[g_py]
        g_pz_rl = obs[l_pz] - obs[g_pz]
        g_vx_rl = obs[l_vx] - obs[g_vx]
        g_vy_rl = obs[l_vy] - obs[g_vy]
        g_vz_rl = obs[l_vz] - obs[g_vz]

        g_p_rl = (g_px_rl**2 + g_py_rl**2 + g_pz_rl**2)**0.5
        g_v_rl = (g_vx_rl**2 + g_vy_rl**2 + g_vz_rl**2)**0.5

        # Magnitude of relative distances and velocities between crafts
        b_p_rg = (b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        b_v_rg = (b_vx_rg**2 + b_vy_rg**2 + b_vz_rg**2)**0.5

        b_p_rl = (b_px_rl**2 + b_py_rl**2 + b_pz_rl**2)**0.5
        b_v_rl = (b_vx_rl**2 + b_vy_rl**2 + b_vz_rl**2)**0.5

        g_p_rl = (g_px_rl**2 + g_py_rl**2 + g_pz_rl**2)**0.5
        g_v_rl = (g_vx_rl**2 + g_vy_rl**2 + g_vz_rl**2)**0.5

        mass_b = 7025
        mass_g = 7025

        acc_max_b = 8000/mass_b
        acc_max_g = 8000/mass_g

        acc_max_b_x = acc_max_b * b_px_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        acc_max_b_y = acc_max_b * b_py_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        acc_max_b_z = acc_max_b * b_pz_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5

        acc_max_g_x = acc_max_b * b_px_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        acc_max_g_y = acc_max_b * b_py_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5
        acc_max_g_z = acc_max_b * b_pz_rg/(b_px_rg**2 + b_py_rg**2 + b_pz_rg**2)**0.5

        t_guard_to_bandit_x = (-b_vx_rg + (b_vx_rg**2 + 2*acc_max_g_x * b_px_rg)**0.5)/acc_max_g_x
        t_guard_to_bandit_y = (-b_vy_rg + (b_vy_rg**2 + 2*acc_max_g_y * b_py_rg)**0.5)/acc_max_g_y
        t_guard_to_bandit_z = (-b_vz_rg + (b_vz_rg**2 + 2*acc_max_g_z * b_pz_rg)**0.5)/acc_max_g_z

        t_guard_to_bandit = max(t_guard_to_bandit_x, t_guard_to_bandit_y, t_guard_to_bandit_z)

        t_lady_to_bandit_x = (-b_vx_rl + (b_vx_rl**2 + 2*acc_max_b_x * b_px_rl)**0.5)/acc_max_b_x
        t_lady_to_bandit_y = (-b_vy_rl + (b_vy_rl**2 + 2*acc_max_b_y * b_py_rl)**0.5)/acc_max_b_y
        t_lady_to_bandit_z = (-b_vz_rl + (b_vz_rl**2 + 2*acc_max_b_z * b_pz_rl)**0.5)/acc_max_b_z

        t_lady_to_bandit = max(t_lady_to_bandit_x, t_lady_to_bandit_y, t_lady_to_bandit_z)

        print("Time from Guard to Bandit: ")
        print(t_guard_to_bandit)

        print("Time from Lady to Bandit: ")
        print(t_lady_to_bandit)


        # Select action based on if the lady or guard is closer to the bandit
        if(t_lady_to_bandit < 80):
            if(t_lady_to_bandit > t_guard_to_bandit):
                scaled_act = [b_px_rl - 200, b_py_rl, b_pz_rl]/np.linalg.norm([b_px_rl - 200, b_py_rl, b_pz_rl])
            else:
                scaled_act = [b_px_rl, b_py_rl, b_pz_rl]/np.linalg.norm([b_px_rl, b_py_rl, b_pz_rl])
        else:
            if(t_lady_to_bandit > t_guard_to_bandit):
               scaled_act = [b_px_rl - 200, 0.8*b_py_rl, 0.9*b_pz_rl]/np.linalg.norm([b_px_rl - 200, b_py_rl, b_pz_rl])
            else:
               scaled_act = [b_px_rl, 0.8*b_py_rl, 0.9*b_pz_rl]/np.linalg.norm([b_px_rl, b_py_rl, b_pz_rl])

        # Prevent speed from increasing when close to the lady
        if b_vx_rl >= 50:
            scaled_act[1] = 0

        # Print the time of convergence
        if b_p_rl < 400:
            print("Time to get within 400m:", (datetime.datetime.now() - start_time))

        # Send the action to KSP to execute
        return {"burn_vec": [scaled_act[0], scaled_act[1], scaled_act[2], 1.0], "vec_type": 0, "ref_frame": 1}

start_time = datetime.datetime.now()

# Call the agent model in KSP
if __name__ == "__main__":
    naive_agent = NaivePursuitAgent()    
    runner = AgentEnvRunner(
        agent=naive_agent, 
        env_cls=LBG1_LG3_I2_V1, 
        env_kwargs=None,
        runner_timeout=100,     # agent runner that will timeout after 100 seconds
        debug=True)
    print(runner.run())   ## Create and initialize the game environment
