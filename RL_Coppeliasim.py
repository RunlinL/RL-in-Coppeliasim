import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sim  # CoppeliaSim's remote API
from stable_baselines3 import DDPG
import time
from stable_baselines3.common.callbacks import BaseCallback

class Robot():

    def __init__(self, frame_name, motor_names=[], client_id=0):
        # If there is an existing connection
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = self.open_connection()

        self.motors = self._get_handlers(motor_names)

        # Robot frame
        self.frame = self._get_handler(frame_name)

    def open_connection(self):
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

        if self.client_id != -1:
            print('Robot connected')
        else:
            print('Connection failed')
        return self.client_id

    def close_connection(self):
        sim.simxGetPingTime(
            self.client_id)  # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
        sim.simxFinish(self.client_id)  # Now close the connection to CoppeliaSim:
        print('Connection closed')

    def isConnected(self):
        c, result = sim.simxGetPingTime(self.client_id)
        # Return true if the robot is connected
        return result > 0

    def _get_handler(self, name):
        err_code, handler = sim.simxGetObjectHandle(self.client_id, name, sim.simx_opmode_blocking)
        return handler

    def _get_handlers(self, names):
        handlers = []
        for name in names:
            handler = self._get_handler(name)
            handlers.append(handler)

        return handlers


    def set_position(self, position, relative_object=-1):
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        sim.simxSetObjectPosition(self.client_id, self.frame, relative_object, position, sim.simx_opmode_blocking)

    def simtime(self):
        return sim.simxGetLastCmdTime(self.client_id)

    def get_position(self, relative_object=-1):
        # Get position relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, position = sim.simxGetObjectPosition(self.client_id, self.frame, relative_object, sim.simx_opmode_blocking)
        return np.array(position)

    def get_velocity(self, relative_object=-1):
        # Get velocity relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, velocity, omega = sim.simxGetObjectVelocity(self.client_id, self.frame, sim.simx_opmode_blocking)
        return np.array(velocity), np.array(omega)

    def get_object_position(self, object_name):
        # Get Object position in the world frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, -1, sim.simx_opmode_blocking)
        return np.array(position)

    def get_object_relative_position(self, object_name):
        # Get Object position in the robot frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, self.frame, sim.simx_opmode_blocking)
        return np.array(position)

    def set_float(self, f, signal='f'):
        return sim.simxSetFloatSignal(self.client_id, signal, f, sim.simx_opmode_oneshot_wait)

    def set_servo_forces(self, servo_angle1, servo_angle2, force_motor1, force_motor2):
        self.set_float(force_motor1, 'f1')  # Force motor 1
        self.set_float(force_motor2, 'f2')  # Force motor 2
        self.set_float(servo_angle1, 't1')  # Servo 1
        self.set_float(servo_angle2, 't2')  # Servo 2

    def get_orientation(self):
        res, orientation = sim.simxGetObjectOrientation(self.client_id, self.frame, -1, sim.simx_opmode_blocking)
        return np.array(orientation)

    def set_orientation(self, orientation):
        sim.simxSetObjectOrientation(self.client_id, self.frame, -1, orientation, sim.simx_opmode_blocking)

class DroneNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.max_steps = 50  # 设定最大步数
        self.current_step = 0

        super(DroneNavigationEnv, self).__init__()

        self.action_space = spaces.Box(low=np.array([np.pi/3, -np.pi/2, 1, 1]), high=np.array([np.pi/2, -np.pi/3, 5, 5]), shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Initialize the robot (drone)
        self.drone = Robot(frame_name='Drone', motor_names=[''], client_id=0)
        self.goal_position = np.array([0, 0, 0.7])  # Define the goal position

        self.initial_position = self.drone.get_position()
        self.initial_orientation = self.drone.get_orientation()

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False  # 或者保持现有的完成逻辑

        # Existing code to apply action
        servo_angle1, servo_angle2, force_motor1, force_motor2 = action
        self.drone.set_servo_forces(servo_angle1, servo_angle2, force_motor1, force_motor2)
        drone_position = self.drone.get_object_position('bicopterBody')
        print(drone_position)
        # Calculate distance to the goal
        distance_to_goal = np.linalg.norm(self.goal_position - drone_position)
        # print(distance_to_goal)

        # Reward function refinement
        if distance_to_goal < 0.1:
            reward = 100  # Large positive reward for reaching the goal
        else:
            reward = -1 * distance_to_goal  # Negative reward based on distance

        # Adjusted termination condition
        done = distance_to_goal < 0.1   # Example condition
        print(f"Current Step Reward: {reward}")
        # Return the step information
        return drone_position, reward, done, False, {}

    def reset(self, seed=None):
        # Reset the drone's position and orientation
        self.drone.set_position(self.initial_position)
        self.drone.set_orientation(self.initial_orientation)

        # Get the initial observation
        initial_observation = self.drone.get_position()
        self.current_step = 0
        return initial_observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.drone.close_connection()

class RewardLogger(BaseCallback):
    def __init__(self, check_freq):
        super(RewardLogger, self).__init__()
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        episode_reward = np.sum(self.locals['rewards'])
        self.episode_rewards.append(episode_reward)
        print(f"Step: {self.n_calls}, Episode Reward: {episode_reward}")
        return True

# Create the Gym environment
env = DroneNavigationEnv()

# Initialize the PPO model
model = DDPG("MlpPolicy", env, verbose=1)

if __name__ == "__main__":

    # Train the model
    reward_logger = RewardLogger(check_freq=1000)
    model.learn(total_timesteps=100, log_interval=10, callback=reward_logger)
    # model.learn(total_timesteps=50000, log_interval=10)

    # Save the model
    model.save("DDPG_drone_navigation")

    # Load and test the trained model
    model = DDPG.load("DDPG_drone_navigation")

    # Test the trained model
    for episode in range(10):
        print(episode)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode} finished with total reward: {total_reward}")
        
    env.close()

