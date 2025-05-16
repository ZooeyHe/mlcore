"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-03-19
"""

import os
import random
from datetime import datetime
import re
from typing import Union
import uuid

import mlcore.file as file
import mlcore.folder as folder
from mlcore.words import ANIMALS, VERBS, ADJECTIVES


"""
String Operations. Functions for generating, parsing, searching, and manipulating strings.
"""


STANDARD_DT_STR_FMT = r"%Y%m%d-%H%M%S"


def is_name(name: str) -> bool:
    """
    Check if a candidate string is a unique name string.

    Args:
        name: the candidate string.

    Returns:
        is_name: a boolean indicating if the candidate string is a standard unique name.
    """
    splits = name.split("_")
    if len(splits) < 2 or len(splits) > 3:
        return False
    # See if the first split contains a dt str.
    try:
        parse_dt_str(splits[0])
    except:
        return False
    # See if the second split is a uuid.
    if not is_uuid(splits[1]) and not is_readable_uuid(splits[1]):
        return False
    # Check for tag validity.
    if len(splits) == 3 and len(splits[2]) == 0:
        return False
    return True


def create_name(
        dt_obj: datetime = None, uuid_str: str = None, tag: str = None, readable: bool = False
    ) -> str:
    """
    Create a unique name string using the timestamp and an UUID string.

    Args:
        dt: A datetime object to use for the timestamp string component of the name.
        uuid: If given, use the given uuid.
        tag: An extra str tag to add to the end of the name.
        readable: whether to use the human-readable uuid

    Return:
        name: A unique name, such as: 20240319-220721_e92e5b34-7ca0-411b-8ca1-fb662fd79ae9_sometag
            If readable, a unique name such as 20240319-220721_cool-yak_sometag
    """
    dt_str = create_dt_str(dt_obj)
    if not uuid_str:
        uuid_str = generate_readable_uuid() if readable else str(uuid.uuid4())
    assert is_uuid(uuid_str) or is_readable_uuid(uuid_str), f"uuid_str is not a uuid."
    name = dt_str + "_" + uuid_str
    if tag:
        if "_" in tag:
            raise ValueError("The tag cannot contain '_'.")
        elif len(tag) == 0:
            raise ValueError("The tag must be at least one character long.")
        return name + "_" + str(tag)
    else:
        return name
    

def parse_name(name: str) -> Union[str,str,str]:
    """
    Given a standard unique name (generated using `create_name`), parse the information.
    
    Args:
        name: The name to parse.

    Returns:
        dt: The datetime object created from the string.
        uuid: the uuid string.
        tag: An optional tag.
    """
    splits = name.split("_")[:3]
    dt_str, uuid_str = splits[:2]
    if len(splits) == 3:
        tag_str = splits[2]
    else:
        tag_str = ""
    dt_obj = parse_dt_str(dt_str)
    return dt_obj, uuid_str, tag_str


def generate_readable_uuid(n_words: int = 2) -> str:
    """
    Generate a readable uuid using common english words.

    Args:
        n_words: can be either 1, 2, or 3. 
            - If 1, returns a noun (animal)
            - If 2, returns a adjective-noun.
            - If 3, returns a verb-adjective-noun.
    
    Return:
        readable_uuid: Something like "running-cool-aardvark", "genius-bear", "whale"
    """
    if n_words == 1:
        return random.choice(ANIMALS)
    elif n_words == 2:
        return random.choice(ADJECTIVES) + "-" + random.choice(ANIMALS)
    elif n_words == 3:
        return random.choice(VERBS) + "-" + random.choice(ADJECTIVES) + "-" + random.choice(ANIMALS)
    else:
        raise ValueError("n_words can only be 1, 2, or 3")


def is_readable_uuid(name: str) -> bool:
    """
    Checks if the string is a readable uuid.
    """
    words = name.split("-")
    if len(words) == 1:
        return bool( words[0] in ANIMALS)
    elif len(words) == 2:
        return bool( words[1] in ANIMALS and words[0] in ADJECTIVES)
    elif len(words) == 3:
        return bool( words[2] in ANIMALS and words[1] in ADJECTIVES and words[0] in VERBS)
    else:
        return False


def get_indexed_name(
        root: str, stem: str = "file", ind_fmt: str = "{:04d}", ext: str = None, abs: bool = True
    ) -> str:
    """
    If a given folder contains files/folders that are named with an increasing index, return a new
    file/folder name that is one higher than the highest existing index.

    For example, if a given folder contains files "file_1.txt", "file_2.txt", and "file_4.txt",
    return "file_5.txt"

    Args:
        root: the given folder containing the index files / folders.
        stem: the stem used for naming this series of indexed files.
        ind_fmt: the formatting of the index integers themselves.
        ext: the extension for the file series. For example: ".txt"
        abs: whether to return the path to the new file or not.
    """
    if ext:
        ext_ = ext[1:]  # Remove the leading "."
        regex = r"^" + stem + r"_([0-9]{1,4})\." + ext_ + r"$"
    else:
        ext = ""
        regex = r"^" + stem + r"_([0-9]{1,4})$"
    # Find all existing names in this series.
    existing = (
        file.find_files_in(root, regex, abs=False) + folder.find_folders_in(root, regex, abs=False)
    )
    # Find the next available index.
    max_ind = max( int(re.search(regex, name).group(1)) for name in existing ) if existing else -1
    next_ind = max_ind + 1

    # Create a new name using the next index.
    filename = str( stem + "_" + ind_fmt.format(next_ind) + ext )
    if abs:
        return os.path.join(root, filename)
    else:
        return filename


def is_uuid(string: str):
    """
    Given a string, check to see if it is a uuid.
    """
    try:
        uuid.UUID(string)
    except ValueError:
        return False
    else:
        return True


def create_dt_str(dt_obj: datetime = None):
    """
    Create a dt_str in the mlcore standard dt formatting.

    Args:
        dt_obj: if given use this datetime object's value for dt_str. Otherwise, use now()
    
    Return:
        dt_str: a datetime string. Example: 20240319-220721
    """
    return (dt_obj or datetime.now()).strftime(STANDARD_DT_STR_FMT)


def parse_dt_str(dt_str: str) -> datetime:
    """
    Given a datetime string using the mlcore standard formatting, return the datetime obj.

    Args:
        dt_str: The datetime string.

    Returns:
        dt_obj: the datetime object created from the string.
    """
    return datetime.strptime(dt_str, STANDARD_DT_STR_FMT)
