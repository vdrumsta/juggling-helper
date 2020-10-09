import cv2
import time
from typing import Tuple, OrderedDict
from dataclasses import dataclass
from collections import OrderedDict
from audio_manager import SoundFactory, Sound

@dataclass
class JuggleDetails:
    """ Used to record how high a juggling ball is thrown and whether it started
        falling down"""
    centroid: Tuple[int, int] # X, Y coordinates of the juggling ball at its heighest point
    starting_time: float
    is_falling: bool = False
    
    @property
    def max_height(self) -> int:
        return self.centroid[1]

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

    def __init__(self, success_area_y, success_area_length, frame_width, reacquisition_time):
        self.drawn_height_points: OrderedDict[int, DrawPoint] = OrderedDict()
        self.successes = 0
        self.failures = 0
        self.reacquisition_time = reacquisition_time

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
        # All the tracked balls will be held here to keep track of whether
        # the ball is travelling upwards or downwards
        self.recorded_balls: OrderedDict[int, JuggleDetails] = OrderedDict()

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
        """ Draws a counter and a percentage of successful throws """
        # Calculate success percentage and check that there's no div by 0
        success_percentage = self.successes / self.failures * 100 if self.failures else 0
        text = f"{int(success_percentage)}% = {self.successes} / {self.failures + self.successes}"
        cv2.putText(frame, text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (132, 224, 190), 2)

    def evaluate_height(self, ball_id, recorded_ball: JuggleDetails):
        """ Checks whether the ball was thrown to the correct height and
            creates a successful/unsuccessful draw point """
        is_successful = self.is_successful_throw(recorded_ball.max_height)
        if is_successful:
            self.successes += 1
            self.success_sound.play()
        else:
            self.failures += 1
            self.failure_sound.play()

        # Record a draw point
        self.drawn_height_points[ball_id] = DrawPoint(
            tuple(recorded_ball.centroid), is_successful, starting_time = time.time()
        )

    def check_for_expired_balls(self, current_balls: OrderedDict):
        """ Check for balls that 'expired' past their reacquisition time.
            This means it is too late to reacquire it and can be considered
            As if it reached its max height """
        for ball_id, recorded_ball in self.recorded_balls.items():
            if not recorded_ball.is_falling:
                time_since_thrown = time.time() - recorded_ball.starting_time

                # Ball has 'expired'
                if time_since_thrown > self.reacquisition_time:
                    recorded_ball.is_falling = True
                    self.evaluate_height(ball_id, recorded_ball)


    def update(self, frame, current_balls: OrderedDict):
        """ Main function that is called every frame to check whether 
            balls are falling and draw success/fail points """
        # Loop through balls and check if they started falling
        for ball_id, centroid in current_balls.items():
            # If the ball was previously recorded
            if ball_id in self.recorded_balls:
                previously_recorded_details = self.recorded_balls[ball_id]

                # Check if the ball is lower than previously recorded. This means 
                # that the ball has reached has reached its max height in the 
                # previous frame
                # NOTE: if the ball is closer to the ground, it's y-coord number will be HIGHER
                if (not previously_recorded_details.is_falling 
                        and previously_recorded_details.max_height < centroid[1]):
                    self.recorded_balls[ball_id].is_falling = True
                    self.evaluate_height(ball_id, self.recorded_balls[ball_id])

                # The ball is still travelling upwards
                else:
                    self.recorded_balls[ball_id].centroid = centroid
            # If the ball hasnt been previously detected, keep a record of it for the future
            else:
                new_recorded_ball = JuggleDetails(
                    centroid = centroid, 
                    starting_time = time.time()
                )
                self.recorded_balls[ball_id] = new_recorded_ball

        self.check_for_expired_balls(current_balls)

        self.draw_recorded_points(frame)
        self.draw_success_counters(frame)
