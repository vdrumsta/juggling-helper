import cv2

class HeightChecker:
    """ This class is used to define a boundary inside which the tracked
        objects that are travelling upwards should stop """

    def __init__(self, starting_y, starting_height, frame_width):
        self.start_y = int(starting_y)
        self.length = int(starting_height)
        self.frame_width = int(frame_width)

    def draw_boundary(self, frame):
        """ Draw a rectangle that represents where juggling balls 
            should be at their heighest """
        end_y = self.start_y + self.length
        
        # setting start_x as -1 prevents the left edge line from being drawn
        cv2.rectangle(frame, (-1, self.start_y), (self.frame_width, end_y), (255, 255, 255), 1)

    def change_start_y(self, amount: int = 2):
        """ Raise or lower height boundary """
        self.start_y += amount

    def change_length(self, amount: int = 2):
        """ Increase or decrease length of the boundary """
        new_length = self.length + amount

        if new_length >= 0:
            self.length = new_length
        else:
            self.length = 0