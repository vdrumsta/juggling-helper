import cv2
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Tuple

@dataclass
class JuggleDetails:
    """ Used to record how high a juggling ball is thrown and whether it started
        falling down"""
    max_height: int
    is_falling: bool = False

@dataclass
class RecordedPoint:
    """ Used to record a success/fail point of where a ball has reached max height.
        Also records the time when it started drawing it so that the point 
        disappears after a certain amount of time. """
    centroid: Tuple[int, int]
    starting_time: float

class HeightChecker:
    """ This class is used to define a boundary inside which the tracked
        objects that are travelling upwards should stop"""

    def __init__(self, starting_y, starting_height, frame_width):
        self.drawn_height_points: OrderedDict[int, RecordedPoint] = OrderedDict()

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

    def draw_max_height_point(self, frame, object_id: int, centroid: Tuple[int, int]):
        """ Draw a success/fail point depending on whether it falls within
            height boundary or not """
        # Check if the point has been drawn before
        if object_id in self.drawn_height_points:
            time_since_drawn = time.time() - self.drawn_height_points[object_id].starting_time

            # if it's been longer than 0.5 secs, then don't draw it anymore
            if time_since_drawn > 0.5:
                del self.drawn_height_points[object_id]
                return
        else:
            # Record that we've drawn a point
            self.drawn_height_points[object_id] = RecordedPoint(
                centroid = centroid, starting_time = time.time()
            )

        # Draw a green point if centroid is within the boundary box
        if centroid[1] >= self.start_y and centroid[1] <= (self.start_y + self.length):
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # Draw a red point if centroid is outside the boundary box
        else:
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

    def update(self, frame, current_balls: OrderedDict):
        """ Main function that should be called every fram to check whether 
            balls are falling and draw success/fail points """

        # Keep track of which previously drawn points to not draw anymore
        points_to_remove = []
        
        # Draw existing height points
        for object_id, recorded_point in self.drawn_height_points.items():
            time_since_drawn = time.time() - recorded_point.starting_time

            # if it's been longer than 0.5 secs since the last time this point 
            # was drawn, then don't draw it anymore
            if time_since_drawn > 0.5:
                points_to_remove.append(object_id)
                continue

            self.draw_max_height_point(frame, object_id, recorded_point.centroid)
        
        # Remove points that have already been drawn for a while
        for id in points_to_remove:
            del self.drawn_height_points[id]

        for object_id, centroid in current_balls.items():
            # If the ball was previously recorded
            if object_id in self.recorded_balls:
                previously_recorded_details = self.recorded_balls[object_id]

                # Check if the ball is lower than previously recorded
                # NOTE: if the ball is closer to the ground, it's y-coord number will be HIGHER
                if (not previously_recorded_details.is_falling 
                        and previously_recorded_details.max_height < centroid[1]):
                    self.recorded_balls[object_id].is_falling = True

                    # Draw a success/fail point on frame
                    self.draw_max_height_point(frame, object_id, centroid)
                # Record the balls current height
                else:
                    self.recorded_balls[object_id].max_height = centroid[1]
            # If the ball hasnt been previously detected, record its height
            else:
                self.recorded_balls[object_id] = JuggleDetails(max_height = centroid[1])
