"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-04
"""

import os
import subprocess
from typing import Tuple, Dict


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


def read_run_cmd() -> str:
    """
    Return the command line command which executed this program.

    Returns:
        cmd: the command which called this python software. For example, if we ran a program like
            `python hello.py arg1 arg2` and hello.py contained `cmd = read_run_cmd()`, the cmd
            variable would contain `python hello.py arg1 arg2`.
    """
    # Get the exact command used to run this script on Linux
    with open('/proc/self/cmdline', 'r') as f:
        terminal_str = f.read().replace('\x00', ' ').strip()
    return terminal_str


def get_environment() -> Dict[str, str]:
    """
    Return a dictionary of the current environmental variables mapped to its values.

    Returns:
        environ_vars: Dictionary of environmental variables and its values.
    """
    decoded_data = {}
    for bkey, bval in os.environ._data.items():
        key, val = bkey.decode("utf-8"), bval.decode("utf-8")
        decoded_data[key] = val
    return decoded_data