import numpy as np
from physics_sim import PhysicsSim
from typing import Callable, Union

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_trajectory:Union[type(None), Callable[[float], np.array]]=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_trajectory: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_trajectory = target_trajectory if target_trajectory is not None else lambda t: np.array([0., 0., 10.])
        
    def get_reward(self, rotor_speeds: np.array):
        """Uses current pose of sim to return reward."""
        epsilon = 1e-4
        position_loss = np.sqrt(np.sum(np.square(self.sim.pose[:3] - self.target_trajectory(self.sim.time))))
        rotor_speed_loss = np.sqrt(np.sum(np.square(rotor_speeds - np.array([404] * 4))))
        tilt_loss = np.square(self.sim.pose[4])
        z_vel_loss = np.square(self.sim.v[2])
        altitude_loss = np.square(self.sim.pose[2] - 10)
        reward = 1e1*np.reciprocal(position_loss + epsilon) + 1e-1*np.reciprocal(rotor_speed_loss + epsilon) + 1e-1*np.reciprocal(altitude_loss + epsilon) # - 1e-2*tilt_loss - 1e-2*z_vel_loss
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds) 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state