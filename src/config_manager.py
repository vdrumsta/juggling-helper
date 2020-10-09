import argparse
import sys
from configparser import ConfigParser
from dataclasses import dataclass

@dataclass
class UserSettings:
    """ Class for holding user configurable settings."""
    scale: float
    frame_width: int
    frame_height: int
    debug: bool
    tracktime: float
    trackrange: int
    success_area_y: int
    success_area_length: int

class ConfigManager:
    """ Configuration manager which holds all functions related
        to user configurable settings 
    """
    default_settings = UserSettings(
        scale = 0.4,
        frame_width = 680,
        frame_height = 480,
        debug = False,
        tracktime = 0.2,
        trackrange = 150,
        success_area_y = int(680 / 4 * 0.4),
        success_area_length = int(480 / 10 * 0.4)
    )

    def __init__(self):
        self.parse_args()

        # Read settings from ini file
        self._config = ConfigParser()
        # User either requested to reset settings or couldn't read config.ini file
        if self._args.reset or not self._config.read('config.ini'):
            self._settings = ConfigManager.default_settings
        else:
            frame_width = int(self._config.get('settings', 'frame_width'))
            frame_height = int(self._config.get('settings', 'frame_height'))
            success_area_y = int(self._config.get('settings', 'success_area_y'))
            success_area_length = int(self._config.get('settings', 'success_area_length'))

            self._settings = UserSettings(
                scale = self._args.scale,
                frame_width = frame_width,
                frame_height = frame_height,
                debug = self._args.debug,
                tracktime = self._args.tracktime,
                trackrange = self._args.trackrange,
                success_area_y = success_area_y,
                success_area_length = success_area_length,
            )

    def parse_args(self):
        """ Read in user passed command line arguments """
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Record the max height at which juggling balls are thrown")
        parser.add_argument('-s', '--scale', type=float, default=0.4, metavar="", help="Resize the window size")
        parser.add_argument('-d', '--debug', action='store_true', help="Enable printing of FPS and draw squares around balls")
        parser.add_argument('-t', '--tracktime', type=float, default=0.2, metavar="", help="Max time to reacquire a tracked ball")
        parser.add_argument('-r', '--trackrange', type=int, default=150, metavar="", help="Max range to reacquire a tracked ball")
        parser.add_argument('--reset', action='store_true', help="Reset settings to their default parameters")
        
        self._args = parser.parse_args()

    def get_settings(self):
        """ Return command line arguments """
        return self._settings

    def set_settings(self, settings: UserSettings):
        """ Write user settings to config.ini file """
        if not self._config.has_section('settings'):
            self._config.add_section('settings')

        self._config.set('settings', 'frame_width', str(settings.frame_width))
        self._config.set('settings', 'frame_height', str(settings.frame_height))
        self._config.set('settings', 'success_area_y', str(settings.success_area_y))
        self._config.set('settings', 'success_area_length', str(settings.success_area_length))

        with open('config.ini', 'w') as f: 
            self._config.write(f)