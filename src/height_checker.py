import cv2

class HeightChecker:
    """ This class is used to define a boundary inside which the tracked
        objects that are travelling upwards should stop """

    def __init__(self, y_coord, height):
        self.cv2 = cv2
        self.y_coord = y_coord
        self.height = height

    def draw_boundary(self, frame):
        pass
