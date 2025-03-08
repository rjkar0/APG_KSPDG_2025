import numpy as np
from Controllers.pid_lbg import PIDController
from kspdg import LBG1_LG3_I2_V1
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
import datetime

# Class to define the agent, here being the PID controller
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

        t, mb, mbf, b_px, b_py, b_pz, b_vx, b_vy, b_vz, l_px, l_py, l_pz, l_vx, l_vy, l_vz, g_px, g_py, g_pz, g_vx, g_vy, g_vz = range(21)


        # Setup of PID controller gain values
        kp = np.array([0.15, 0.15, 0.1,   # Bandit position relative to lady
                        0, 0, 0,    # Bandit velocity relative to lady
                        0.2, 0, 0,    # Bandit position relative to guard
                        0, 0, 0])   # Bandit velocity relative to guard

        ki = np.array([0, 0, 0,     # Naming scheme is consistent
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0])

        kd = np.array([0.6, 0.6, 0.6,     # Naming scheme is consistent
                        0, 0, 0,
                        0, 0, 0,
                        0, 0, 0])
        pid = PIDController(kp, ki, kd,error_dim=12)
        dt = 0


        # Calculate time delta
        dt = obs[t] - dt

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


        # Form error state and compute PID control output
        err_state = np.array([b_px_rl, b_py_rl, b_pz_rl, b_vx_rl, b_vy_rl, b_vz_rl, b_px_rg - 500, b_py_rg, b_pz_rg, b_vx_rg, b_vy_rg, b_vz_rg])
        control_output = pid.compute_control_output(err_state,dt)

        # Scale action based on control output
        scaled_act = [control_output[0], control_output[1], control_output[2]]/np.linalg.norm([control_output[0], control_output[1], control_output[2]])
        # act = [scaled_act[0],scaled_act[1],scaled_act[2], 0.5]

        # Prevent speed from increasing when close to the lady
        if b_vx_rl >= 50:
            scaled_act[1] = 0

        # Print the time of convergence
        if b_p_rl < 400:
            print("Time to get within 400m:", (datetime.datetime.now() - start_time))

        # Send the action to KSP to execute
        return {"burn_vec": [scaled_act[0], scaled_act[1], scaled_act[2], 1.0], "vec_type": 0, "ref_frame": 1}

start_time = datetime.datetime.now()

if __name__ == "__main__":
    naive_agent = NaivePursuitAgent()    
    runner = AgentEnvRunner(
        agent=naive_agent, 
        env_cls=LBG1_LG3_I2_V1, 
        env_kwargs=None,
        runner_timeout=100,     # agent runner that will timeout after 100 seconds
        debug=True)
    print(runner.run())   ## Create and initialize the game environment
