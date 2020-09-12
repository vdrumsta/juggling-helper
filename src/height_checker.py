import cv2
import time
from typing import Tuple
from playsound import playsound
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class JuggleDetails:
    """ Used to record how high a juggling ball is thrown and whether it started
        falling down"""
    max_height: int
    is_falling: bool = False

@dataclass
class DrawPoint:
    """ Used to record a success/fail point of where a ball has reached max height.
        Also records the time when it started drawing it so that the point 
        disappears after a certain amount of time. """
    centroid: Tuple[int, int]
    is_successful: bool
    starting_time: float

class HeightChecker:
    """ This class is used to define a boundary inside which the tracked
        objects that are travelling upwards should stop"""

    def __init__(self, starting_y, starting_height, frame_width):
        self.drawn_height_points: OrderedDict[int, DrawPoint] = OrderedDict()
        self.successes = 0
        self.failures = 0

        # Starting y coordinate for drawing the height boundary box
        self.start_y = int(starting_y)
        # The vertical length of the boundary box
        self.length = int(starting_height)
        # The horizontal width of the boundary box
        self.frame_width = int(frame_width)
        # All the tracked balls will be held here to keep track of whether
        # the ball is travelling upwards or downwards
        self.recorded_balls: OrderedDict[int, JuggleDetails] = OrderedDict()

    def draw_boundary(self, frame):
        """ Draw a rectangle that represents where juggling balls 
            should be at their heighest"""
        end_y = self.start_y + self.length
        
        # setting start_x as -1 prevents the left edge line from being drawn
        cv2.rectangle(frame, (-1, self.start_y), (self.frame_width, end_y), (255, 255, 255), 1)

    def change_start_y(self, amount: int = 2):
        """ Raise or lower height boundary"""
        self.start_y += amount

    def change_boundary_length(self, amount: int = 2):
        """ Increase or decrease length of the boundary"""
        new_length = self.length + amount

        if new_length >= 0:
            self.length = new_length
        else:
            self.length = 0

    def is_successful_throw(self, centroid: Tuple[int, int]):
        """ Returns true if the ball's max height was inside a boundary
            box (success) and false if its max height was outside the
            boundary box (failure) """
        # Check if max height is inside the boundary box
        if centroid[1] >= self.start_y and centroid[1] <= (self.start_y + self.length):
            return True
        # Or outside it
        else:
            return False

    def draw_recorded_points(self, frame):
        # Keep track of which previously drawn points to not draw anymore
        points_to_remove = []
        
        # Draw existing height points
        for object_id, recorded_point in self.drawn_height_points.items():
            time_since_drawn = time.time() - recorded_point.starting_time

            # if it's been longer than 0.5 secs since the last time this point 
            # was drawn, then don't draw it anymore
            if time_since_drawn > 0.5:
                points_to_remove.append(object_id)
            else:
                draw_color = (0, 255, 0) if recorded_point.is_successful else (0, 0, 255)
                cv2.circle(frame, recorded_point.centroid, 4, draw_color, -1)
        
        # Remove points that have already been drawn for a while
        for id in points_to_remove:
            del self.drawn_height_points[id]

    def draw_success_counters(self, frame):
        # Calculate success percentage and check that there's no div by 0
        success_percentage = self.successes / self.failures * 100 if self.failures else 0
        text = f"{int(success_percentage)}% = {self.successes} / {self.failures + self.successes}"
        cv2.putText(frame, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (132, 224, 190), 2)

    def update(self, frame, current_balls: OrderedDict):
        """ Main function that should be called every fram to check whether 
            balls are falling and draw success/fail points """
        # Loop through balls and check if they started falling
        for object_id, centroid in current_balls.items():
            # If the ball was previously recorded
            if object_id in self.recorded_balls:
                previously_recorded_details = self.recorded_balls[object_id]

                # Check if the ball is lower than previously recorded. This means 
                # that the ball has reached has reached its max height in the 
                # previous frame
                # NOTE: if the ball is closer to the ground, it's y-coord number will be HIGHER
                if (not previously_recorded_details.is_falling 
                        and previously_recorded_details.max_height < centroid[1]):
                    self.recorded_balls[object_id].is_falling = True

                    is_successful = self.is_successful_throw(centroid)
                    if is_successful:
                        self.successes += 1
                        playsound('correct.wav')
                    else:
                        self.failures += 1
                        playsound('incorrect.wav')

                    # Record a draw point
                    self.drawn_height_points[object_id] = DrawPoint(
                        tuple(centroid), is_successful, starting_time = time.time()
                    )

                # Record the balls current height
                else:
                    self.recorded_balls[object_id].max_height = centroid[1]
            # If the ball hasnt been previously detected, record its height
            else:
                self.recorded_balls[object_id] = JuggleDetails(max_height = centroid[1])

        self.draw_recorded_points(frame)
        self.draw_success_counters(frame)
