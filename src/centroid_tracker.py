# import the necessary packages
import time
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, reacquisitionDist = 50, maxDisappearedTime=0.2):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # Store the time that a given object is allowed to be marked 
        # as "disappeared" until we need to deregister the object from tracking
        self.maxDisappearedTime = maxDisappearedTime

        # Store the maximum distance within which an object can "reacquire"
        # a centroid when it has been marked as "disappeared"
        self.reacquisitionDist = reacquisitionDist

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # Loop over disappeared objects
            for objectID in list(self.disappeared.keys()):
                # If we have reached a maximum time that an object can stay
                # disappeared for, then deregister it
                timeSinceDisappeared = time.time() - self.disappeared[objectID]
                if timeSinceDisappeared > self.maxDisappearedTime:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, there are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Keep a track of indexes to remove from the D array
            rowsToRemove = []
            # Filter out results that are outside centroid's requisition range
            for i, row in enumerate(D):
                # If the centroid has disappeared before calculate its time
                # since it disappeared
                if self.disappeared[objectIDs[i]] > 0:
                    timeSinceDisappeared = time.time() - self.disappeared[objectIDs[i]]
                else:
                    timeSinceDisappeared = 0
                
                # Calculate how much to scale the new reacquisition distance
                reacquisitionDistScalar = 1 - (timeSinceDisappeared / self.maxDisappearedTime)
                # If less than 0, then the object wont be tracked anymore
                if reacquisitionDistScalar <= 0:
                    rowsToRemove.append(i)
                    self.deregister(objectIDs[i])
                    continue

                currentReacquisitionDist = self.reacquisitionDist * reacquisitionDistScalar

                # If any of the distances are grater than the currentReacquisitionDist
                # then set their distance to infinity. This will make sure that we
                # don't consider this distance for reacquisition
                for col in row:
                    col = float('inf') if col > currentReacquisitionDist else col

                isAllInfs = True
                # If the entire row is made up of infs, remove the row
                for col in row:
                    if not col == float('inf'):
                        isAllInfs = False
                        break

                if isAllInfs:
                    rowsToRemove.append(i)

            # Remove rows that will not be considered for reacquisition.
            # We loop backwards so that we don't throw the the subsequent indexes
            for rowToRemove in sorted(rowsToRemove, reverse=True):
                del objectIDs[rowToRemove]
                D = np.delete(D, rowToRemove, 0)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # Grab the object ID for the corresponding row
                    # index and note the time when it disappeared
                    objectID = objectIDs[row]
                    if self.disappeared[objectID] == 0:
                        self.disappeared[objectID] = time.time()

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects