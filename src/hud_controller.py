import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import Tuple, OrderedDict


import cv2
from audio_manager import SoundFactory, Sound

@dataclass
class DrawPoint:
    """ Used to record a success/fail point of where a ball has reached max height.
        Also records the time when it started drawing it so that the point 
        disappears after a certain amount of time.
    """
    draw_centroid: Tuple[int, int]
    starting_time: float
    is_successful: bool

class HUDController:
    """ This class is used to count and mark the juggling balls which reached a
        max height. It also plays the success/fail sound that tells whether it
        reached max height in the desirable boundary box
    """

    def __init__(self, success_area_y, success_area_length, frame_width):
        self.current_draw_points: OrderedDict[int, DrawPoint] = OrderedDict()
        self.drawn_ball_ids = []
        self.not_drawn_balls: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.successes = 0
        self.failures = 0

        # Create success/failure sounds
        sound_factory = SoundFactory()
        self.success_sound = sound_factory.create_audio(seconds = 0.1, frequency = 1000)
        self.failure_sound = sound_factory.create_audio(seconds = 0.1, frequency = 300)

        # Starting y coordinate for drawing the height boundary box
        self.start_y = int(success_area_y)
        # The vertical length of the boundary box
        self.length = int(success_area_length)
        # The horizontal width of the boundary box
        self.frame_width = int(frame_width)

    def draw_ball_points(self, frame, tracked_balls):
        """ Record all newly tracked balls and if the ball started falling,
            then start drawing its max point reached.
            Also draw success/fail statistics and max height box
        """
        for ball_id, ball_value in tracked_balls.items():
            # If this is the first time we are tracking this ball,
            # then note down its id
            if (
                    (not ball_id in self.drawn_ball_ids) and 
                    (not ball_id in self.not_drawn_balls)
                ):
                self.not_drawn_balls[ball_id] = tuple(ball_value.max_height_centroid)

            # Register a draw point if it started falling
            if not ball_id in self.drawn_ball_ids and ball_value.is_falling:
                self.register_draw_point(ball_id, ball_value.max_height_centroid)

        new_not_drawn_balls = {}
        for ball_id, max_height_centroid in self.not_drawn_balls.items():
            # Register a draw point if it has disappeared for too long.
            # To check this we will see if any previously tracked balls is
            # no longer tracked, but hasn't been drawn yet
            if not ball_id in tracked_balls and not ball_id in self.drawn_ball_ids:
                self.register_draw_point(ball_id, max_height_centroid)
            else:
                new_not_drawn_balls[ball_id] = max_height_centroid
        self.not_drawn_balls = new_not_drawn_balls

        self.draw_recorded_points(frame)

    def get_successes(self) -> int:
        return self.successes

    def get_failures(self) -> int:
        return self.failures

    def draw_boundary(self, frame):
        """ Draw a rectangle that represents where juggling balls 
            should be at their heighest"""
        end_y = self.start_y + self.length
        
        # setting start_x as -1 prevents the left edge line from being drawn
        cv2.rectangle(frame, (-1, self.start_y), (self.frame_width, end_y), (0, 255, 0), 1)

    def change_boundary_y_pos(self, amount: int = 2):
        """ Raise or lower height boundary"""
        self.start_y += amount

    def change_boundary_length(self, amount: int = 2):
        """ Increase or decrease length of the boundary"""
        new_length = self.length + amount

        if new_length >= 0:
            self.length = new_length
        else:
            self.length = 0

    def is_successful_throw(self, ball_height: int):
        """ Returns true if the ball's max height was inside a boundary
            box (success) and false if its max height was outside the
            boundary box (failure) """
        # Check if max height is inside the boundary box
        if ball_height >= self.start_y and ball_height <= (self.start_y + self.length):
            return True
        # Or outside it
        else:
            return False
    
    def draw_recorded_points(self, frame):
        """ Draws a green/red dot for successful/unsuccesful throw """
        draw_points_to_remove = []
        
        # Draw existing height points
        for object_id, recorded_point in self.current_draw_points.items():
            time_since_drawn = time.time() - recorded_point.starting_time

            # if it's been longer than 0.5 secs since the last time this point 
            # was drawn, then don't draw it anymore
            if time_since_drawn > 0.5:
                draw_points_to_remove.append(object_id)
            else:
                draw_color = (0, 255, 0) if recorded_point.is_successful else (0, 0, 255)
                cv2.circle(frame, recorded_point.draw_centroid, 4, draw_color, -1)
        
        # Remove points that have already been drawn for a while
        for id in draw_points_to_remove:
            del self.current_draw_points[id]

    def draw_success_counters(self, frame):
        """ Draws a counter and a percentage of successful throws """
        # Calculate success percentage and check that there's no div by 0
        success_percentage = self.successes / (self.failures + self.successes) * 100 if self.failures else 0
        text = f"{int(success_percentage)}% = {self.successes} / {self.failures + self.successes}"
        cv2.putText(frame, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (132, 224, 190), 2)

    def register_draw_point(self, ball_id, centroid):
        """ Checks whether the ball was thrown to the correct height and
            creates a successful/unsuccessful draw point """
        is_successful = self.is_successful_throw(centroid[1])
        if is_successful:
            self.successes += 1
            self.success_sound.play()
        else:
            self.failures += 1
            self.failure_sound.play()

        # Record a draw point
        self.current_draw_points[ball_id] = DrawPoint(
            draw_centroid = centroid,
            starting_time = time.time(),
            is_successful = is_successful
        )
        self.drawn_ball_ids.append(ball_id)
