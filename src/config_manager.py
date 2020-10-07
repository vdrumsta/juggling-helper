import argparse

class ConfigManager:
    """ Configuration manager which holds all related user configurable settings """
    def __init__(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Record the max height at which juggling balls are thrown")
        parser.add_argument("-s", "--scale", type=float, default=0.4, metavar="", help="Resize the window size")
        parser.add_argument("-d", "--debug", action="store_true", help="Enable printing of FPS and draw squares around balls")
        parser.add_argument("-t", "--tracktime", type=float, default=0.2, metavar="", help="Max time to reacquire a tracked ball")
        parser.add_argument("-r", "--range", type=int, default=150, metavar="", help="Max range to reacquire a tracked ball")
        
        self._args = parser.parse_args()

    def get_args(self):
        """ Return command line arguments """
        return self._args
