import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Lane:
    """
    Represents a lane on the road
    """

    def __init__(self, orig_frame):
        self.orig_frame = orig_frame
        self.lane_lane_markings = None