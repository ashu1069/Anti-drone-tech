from typing import no_type_check
import airsim
from collections import deque, namedtuple
import numpy as np 
import time
from PIL import Image

Experience = namedtuple("Experience", field_names=['state', 'reward', 'done', 'new_state'])

class Drone:
    def __init__(self, start_position, velocity_factor, hparams):
        self.hparams = hparams
        self.start_position = start_position
        self.scaling_factor = velocity_factor
        self.client = None
        self.reset()
    
    def initializeClient(self):

        if self.client is None:
            self.client =airsim.MultirotorClient()
            self.client.confirmConnection()
        else:
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
    def hasCollided(self):

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        else:
            return False

    def convertPositionToTensor(self, position):

        current_position = tensor([[position.x_val], [position.y_val],[position.z_val]])
        return current_position

    def get AgentState(self):

        position = self.convertPositionToTensor(self.position.position)
        state_image = self.getImage()
        state_signal_strength = self.sensor.getSignalStrength(position)
        
        state_image=state_image/250

        return{"image": state_image, "signal": state_signa_strength}

    def getAction(self, net, epsilon, device):

        if np.random.random()<epsilon:
            action = np.random.randint(self.hparams.model.actions)
        else:
            state_dict = self.getAgentState()

            action=int(action.item())
        return action

    def playStep(self, net, epsilon, device):

        action = self.getAction(net, epsilon, device)
        action_offset=self.nextAction(action)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        state_dict = self.getAgentState()

        self.client.moveByVelocityAsync(
            quad_vel.x_val + action_offset[0],
            quad_vel.y_val + action_offset[1],
            quad_vel.z_val + action_offset[2]
        ).join()

        current_position = self.convertPositionToTensor(self.position.position)
        done, reward = self.isDone()

        new_state_dict = self.getAgentState()
        self.position = self.client.simGetVehiclePose()
        print(self.position.position)

        if not done:
            reward = self.sensor.get_reward(current_position)
            print(reward)
            exp = Experience(state_dict, action, reward, done, new_state_dict)
            self.buffer.append(exp)

            if done:
                self.reset()
        return reward, done
    
    def GetImage(self):
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        return self.postprocessImage(resposnses)

    def postprocessImage(self, responses):
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(448, 448, 3)
        img2d = np.flipud(img_rgba)

        img_out = img2d.copy
        return image_out
