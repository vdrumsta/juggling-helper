import time
from typing import Tuple, OrderedDict, List
from dataclasses import dataclass
from collections import OrderedDict



class MaxHeightChecker:
    """ This class is used to track juggling balls that reached their max height """

    def __init__(self, reacquisition_range, reacquisition_time):
        self.reacquisition_time = reacquisition_time
        self.reacquisition_range = reacquisition_range

        # All the tracked balls will be held here to keep track of whether
        # the ball is travelling upwards or downwards
        self.recorded_balls: OrderedDict[int, JuggleDetails] = OrderedDict()

    def get_balls_falling_state(self, current_balls: OrderedDict) -> List:
        """ Main function that is called every frame to check whether 
            balls are falling """
        ball_falling_states = []
        
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
                    ball_falling_states.append(True)

                # The ball is still travelling upwards
                else:
                    self.recorded_balls[ball_id].centroid = centroid
                    ball_falling_states.append(False)
            # If the ball hasnt been previously detected, keep a record of it for the future
            else:
                new_recorded_ball = JuggleDetails(
                    centroid = centroid, 
                    starting_time = time.time()
                )
                self.recorded_balls[ball_id] = new_recorded_ball
                ball_falling_states.append(False)

        #self._check_for_expired_balls(current_balls)

        return ball_falling_states

    def _check_for_expired_balls(self, current_balls: OrderedDict):
        """ Check for balls that 'expired' past their reacquisition time.
            This means it is too late to reacquire it and can be considered
            As if it reached its max height """
        for recorded_ball in self.recorded_balls.values():
            if not recorded_ball.is_falling:
                time_since_thrown = time.time() - recorded_ball.starting_time

                # Ball has 'expired'
                if time_since_thrown > self.reacquisition_time:
                    recorded_ball.is_falling = True
