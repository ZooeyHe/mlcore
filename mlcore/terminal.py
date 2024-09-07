"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-04
"""

import os
import subprocess
from typing import Tuple


"""
Utility functions for running and capturing commands from the terminal.
"""


def simple_execute(cmd: str) -> None:
    """
    Execute simple commands on the command line. No output is returned. No accomadating escape
    characters within the cmd.

    Args:
        cmd: the command to run.
    """
    os.system(cmd)


def execute_cmd(cmd: str) -> Tuple[int, str]:
    """
    Given a command, execute it in the terminal, printing to stdout, and capturing the 
    return code and output as a string.

    Args:
        cmd: the command to run.

    Returns:
        exitcode: the return code from the execution. Code 0 is success. Code 1 is error. 
            For more: https://www.baeldung.com/linux/status-codes
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    exitcode = process.wait()
    return exitcode, process.stdout.read()