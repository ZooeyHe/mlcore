"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-04
"""

from typing import Set
from string import Formatter

"""
Operations performed on strings.
"""


def get_string_fields(template: str) -> Set[str]:
    """
    Given a string with keyword fields, get all of the field keywords.

    Example: get_string_fields("{hello} this is an {example}, alright?") -> {"hello", "example"}

    Args:
        template: a string with fieldnames.
    
    Returns:
        field: a list of string fieldnames.
    """
    return { field for _, field, _, _ in Formatter().parse(template) if field }