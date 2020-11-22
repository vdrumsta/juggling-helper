import time
from collections import OrderedDict
from typing import Tuple, OrderedDict, List
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance as dist

from height_checker import MaxHeightChecker

@dataclass
class JuggleDetails:
    """ Used to record how high a juggling ball is thrown and whether it started
        falling down"""
    centroid: Tuple[int, int] # X, Y coordinates of the juggling ball's current point
    max_height_centroid: Tuple[int, int] # X, Y coordinates of the juggling ball at its heighest point
    last_seen_timestamp: float
    is_falling: bool = False

    @property
    def max_height(self) -> int:
        return self.max_height_centroid[1]

class CentroidTracker():
    """ Used to identify and track the same balls between frames """
    def __init__(self, reacquisitionDist = 50, maxDisappearedTime=0.2):
        self.nextObjectID = 0
        self.objects: OrderedDict[int, JuggleDetails] = OrderedDict()

        # Store the time that a given object is allowed to be marked 
        # as "disappeared" until we need to deregister the object from tracking
        self.maxDisappearedTime = maxDisappearedTime

        # Store the maximum distance within which an object can "reacquire"
        # a centroid when it has been marked as "disappeared"
        self.reacquisitionDist = reacquisitionDist

    def register(self, centroid):
        # When registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = JuggleDetails(
            centroid = tuple(centroid),
            max_height_centroid = tuple(centroid),
            last_seen_timestamp = time.time()
        )
        self.nextObjectID += 1

    def remove_disappeared_balls(self):
        """ Go through all the currently tracked objects and remove the ones
            that have not been seen for more than allowed time
        """
        # Build a list of objects that havent disappeared for too long
        new_objects_dict: OrderedDict[int, JuggleDetails] = OrderedDict()

        for ball_key, ball_value in self.objects.items():
            # Check if we have reached a maximum time that an object can stay
            # disappeared for, and if we did then deregister it
            timeSinceDisappeared = time.time() - ball_value.last_seen_timestamp
            if timeSinceDisappeared < self.maxDisappearedTime:
                new_objects_dict[ball_key] = ball_value

        # Override objects dict with a dict that doesnt had disappeared balls
        self.objects = new_objects_dict
    
    def check_if_falling(self, old_centroid, new_centroid):
        """ Check if the ball is lower than previously recorded. This means 
            that the ball has reached has reached its max height in the 
            previous frame. Returns true if the balls is falling.
        """
        # NOTE: if the ball is closer to the ground, it's y-coord number will be HIGHER
        return old_centroid[1] < new_centroid[1]

    def set_objects_centroid(self, objectID: int, centroid: Tuple[int, int]):
        """ Set the objects current centroid position and update
            the object's max height centroid if the new position
            is heigher than the previous max height centroid
        """
        self.objects[objectID].centroid = centroid

        if (centroid[1] < self.objects[objectID].max_height_centroid[1]):
            self.objects[objectID].max_height_centroid = centroid
        

    def update(self, rects):
        self.remove_disappeared_balls()

        # Check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # Return early as there are no centroids or tracking info
            # to update
            return self.objects

        # Initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # Loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # Use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # Otherwise, there are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # Grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = [ball.centroid for ball in self.objects.values()]
            objectFallingStates = [ball.is_falling for ball in self.objects.values()]
            
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Calculate the coordinate difference between the stored object
            # centroids and input centroids. This will tell us whether an
            # input centroid is above/below and left/right of the stored 
            # object centroid
            pos_diff = np.zeros(
                (len(objectCentroids), len(inputCentroids)),
                dtype=np.dtype((np.float32, (2,)))
            )
            for i, _ in enumerate(pos_diff):
                for j, _ in enumerate(pos_diff[i]):
                    pos_diff[i][j] = (
                        objectCentroids[i][0] - inputCentroids[j][0],
                        objectCentroids[i][1] - inputCentroids[j][1]
                    )

            # Filter out results that are outside of object centroid's requisition range
            for i, row in enumerate(D):
                # If any of the distances are grater than the currentReacquisitionDist
                # then set their distance to infinity. This will make sure that we
                # don't consider this distance for reacquisition
                for j, col in enumerate(row):
                    D[i][j] = float('inf') if col > self.reacquisitionDist else col

            # For the objects that are already falling, filter out the results
            # where the input object is travelling up
            for i, row in enumerate(D):
                for j, col in enumerate(row):
                    # If the ball is falling and the input centroid is above object
                    # centroid set their distance to infinity. This will make sure that we
                    # don't consider this distance for reacquisition
                    # NOTE: if the ball is closer to the ground, 
                    # pos_diff[i][j][1] will be positive
                    if objectFallingStates[i] and (pos_diff[i][j][1] > 0):
                        D[i][j] = float('inf')

            # In order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # Next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # In order to determine if we need to update or register
            # centroids, we need to track the noes that we examined
            usedCols = set()
            usedRows = set()

            # Loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # Nf we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # TODO: Check if this is actually helping bug
                # "Tracker reacquires balls that are outside of their range"
                #
                # If the row + col combination has the distance of infinity
                # then ignore it as the object is outside of the
                # reacquisition distance
                if D[row][col] == float('inf'):
                    continue

                # Otherwise, grab the object ID for the current row, set its new
                #  centroid, falling state, and reset the last seen timestamp
                objectID = objectIDs[row]
                self.objects[objectID].is_falling = self.check_if_falling(
                    self.objects[objectID].centroid, inputCentroids[col]
                )
                self.set_objects_centroid(objectID, centroid = tuple(inputCentroids[col]))
                self.objects[objectID].last_seen_timestamp = time.time()

                # Indicate that we have examined each of the row and
                # column indexes, respectively
                usedCols.add(col)
                usedRows.add(row)

            # Compute both the row and column index we have NOT yet
            # examined
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Register all unused input centroids
            for col in unusedCols:
                self.register(inputCentroids[col])

        # Return the set of tracked objects
        return self.objects