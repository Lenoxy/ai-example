from enum import Enum

import numpy as np
from Config import DETAIL_LOGGING


class Utils:
    @staticmethod
    def rgb2gray(img):
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def pre_process(img):
        return np.mean(img[::1, ::1], axis=2).astype(np.uint8)

    @staticmethod
    def format_action(random: bool, action: int):
        if DETAIL_LOGGING and random:
            print("Random action:", Actions(action))
        elif DETAIL_LOGGING and not random:
            print("Predicted action:", Actions(action))



class Actions(Enum):
    NOTHING = 0
    FORWARD = 1
    FORWARD_JUMP = 2
    FORWARD_SPRINT = 3
    FORWARD_JUMP_SPRINT = 4
    JUMP = 5
    BACKWARD = 6
