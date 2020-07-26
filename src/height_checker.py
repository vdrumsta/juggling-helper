import cv2
import time
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class JuggleDetails:
    """ Used to record how high a juggling ball is thrown and whether it started
        falling down"""
    max_height: int
    is_falling: bool = False

    """ Used to record the moment when the y-coord is first recorded to be lower,
        than the last recorded y-coord"""
    recorded_lower_y: bool = False

    """ The time when a lower y-coord was first recorded"""
    lower_y_time: float = 0.0

class HeightChecker:
    """ This class is used to define a boundary inside which the tracked
        objects that are travelling upwards should stop"""

    def __init__(self, starting_y, starting_height, frame_width):
        self.start_y = int(starting_y)
        self.length = int(starting_height)
        self.frame_width = int(frame_width)
        self.recorded_balls = OrderedDict()

    def draw_boundary(self, frame):
        """ Draw a rectangle that represents where juggling balls 
            should be at their heighest"""
        end_y = self.start_y + self.length
        
        # setting start_x as -1 prevents the left edge line from being drawn
        cv2.rectangle(frame, (-1, self.start_y), (self.frame_width, end_y), (255, 255, 255), 1)

    def change_start_y(self, amount: int = 2):
        """ Raise or lower height boundary"""
        self.start_y += amount

    def change_length(self, amount: int = 2):
        """ Increase or decrease length of the boundary"""
        new_length = self.length + amount

        if new_length >= 0:
            self.length = new_length
        else:
            self.length = 0

    def check_if_falling(self, current_balls: OrderedDict):
        """ Loops through all present balls and checks if they are falling"""
        for (objectID, centroid) in current_balls.items():
            # If the ball was previously recorded
            if objectID in self.recorded_balls:
                ball_details = self.recorded_balls[objectID]

                # Check if the ball is lower than previously recorded
                # NOTE: if the ball is closer to the ground, it's y-coord will be HIGHER
                if ball_details.max_height < centroid[1]:
                    if ball_details.recorded_lower_y:
                        # Time that the ball has been lowering for 
                        lowering_time = time.time() - ball_details.lower_y_time
                        if lowering_time > 1.0:
                            print("max height")
                    else:
                        ball_details.recorded_lower_y = True
                        ball_details.lower_y_time = time.time()
                else:
                    self.recorded_balls[objectID].max_height = centroid[1]
            else:
                self.recorded_balls[objectID] = JuggleDetails(max_height = centroid[1])