"""
Author: Zooey He
Email: zhuohonghe@gmail.com
Date: 2023-05-10
"""

from collections import deque
import time
from typing import Union, Tuple

"""
Estimated Time of Completion (ETC) is a timer-like package which also estimates time until the
completion of a task by considering the split times.
"""

# Default format for (converting hours, minutes, seconds) into a string.
DEFAULT_HMS_STR_FORMAT = "{:02d}:{:02d}:{:05.2f}"


class ETC():
    """
    Estimated Time of Completion (ETC) Timer Class.
    """
    def __init__(self, num_it: int, window_size: int = None):
        """
        Constructor for ETC class.

        Args:
            num_it: the total number of iterations to expect for completion.
            window_size: the number of past ticks to use for rate estimation. Default None means
                that we are not using a window (aka calculate rate by using first tick)
        """
        self.last_it = num_it
        self.curr_it = 0
        self.start_ts = time.time()
        self.rate_window = deque([self.start_ts], window_size+1 if window_size else None)

    def tick(self):
        """
        Tick function to use at the end of each iteration.
        """
        self.curr_it += 1
        self.rate_window.append(time.time())
        return self
    
    def etc(self, hms: bool = False, as_str: bool = False) -> Union[float,Tuple[int,int,float],str]:
        """
        Get the expected remaining time until completion.

        Args:
            hms: If True, return tuple representing etc in (n_hours, n_min, n_sec).
                Otherwise, return etc in secs.
            as_str: If True, returns a string instead of a float.
        """
        if self.curr_it == 0:
            raise ValueError("ETC must be ticked at least once before completion estimation")
        if self.curr_it >= self.last_it:
            remaining = 0.0
        else:
            curr_it_ts = self.rate_window[-1]
            win_begin_ts = self.rate_window[0]
            secs_since_tick = time.time() - curr_it_ts
            rate = (curr_it_ts - win_begin_ts) / ( len(self.rate_window) - 1 ) # secs / it
            remaining = (self.last_it - self.curr_it) * rate - secs_since_tick
        if hms:
            h, m, s =  self.secs2hms(remaining)
            if as_str:
                return self.hms2str(h, m, s)
        else:
            if as_str:
                return str(remaining) + "s"
            else:
                return remaining
        
    def elapsed(
            self, hms: bool = False, as_str: bool = False
        ) -> Union[float,Tuple[int,int,float],str]:
        """
        Return the amount of time that has elapsed since the start of this ETC.

        Args:
            hms: If True, return tuple representing etc in (n_hours, n_min, n_sec).
                Otherwise, return etc in secs.
        """
        elapsed = time.time() - self.start_ts
        if hms:
            h, m, s =  self.secs2hms(elapsed)
            if as_str:
                return self.hms2str(h, m, s)
        else:
            if as_str:
                return str(elapsed) + "s"
            else:
                return elapsed
        
    def elapsed_tick(
            self, hms: bool = False, as_str: bool = False
        ) -> Union[float,Tuple[int,int,float],str]:
        """
        Return the amount fo time that has elapsed since the last tick.

        Args:
            hms: If True, return tuple representing etc in (n_hours, n_min, n_sec).
                Otherwise, return etc in secs.
        """
        elapsed = time.time() - self.rate_window[-1]
        if hms:
            h, m, s =  self.secs2hms(elapsed)
            if as_str:
                return self.hms2str(h, m, s)
        else:
            if as_str:
                return str(elapsed) + "s"
            else:
                return elapsed
    
    @property
    def remaining_its(self) -> int:
        """
        Return the remaining number of iterations until completion.
        """
        return self.last_it - self.curr_it

    @staticmethod
    def hms2str(h: int, m: int, s: float, fmt: str = None) -> str:
        """
        Convert h, m, s into a string.
        """
        fmt = fmt or DEFAULT_HMS_STR_FORMAT
        return fmt.format(h, m, s)
    
    @staticmethod
    def secs2hms(delta: float) -> Tuple[int,int,float]:
        """
        Given a time delta (in seconds) return the hours, minutes, and seconds.
        """
        hours = delta // 3600
        minutes = (delta % 3600) // 60
        secs = delta % 60
        return int(hours), int(minutes), float(secs)