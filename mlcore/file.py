"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-07-03
"""

from contextlib import contextmanager
import errno
import json
import mimetypes
import os
import pandas as pd
import re
import shutil
from typing import Dict, List, Optional

import mlcore.folder as folder


"""
Functions relating to file operations.
"""


def read_text(in_filepath: str) -> str:
    """
    Read a text-type document into a string. No file extension checking. Will attempt to read all
    file types into a string.
    """
    contents = ""
    with open(in_filepath, "r") as buffer:
        contents = f"{buffer.read()}"  # Wrap in f-string to allow for special characters.
    return contents


def save_text(text: str, out_filepath: str, mode: str = "w") -> None:
    """
    Save the text string to a filepath.

    Args:
        text: the text to write
        out_filepath: the text file to write to.
        mode: the write mode {w: write (default), a: append}
    """
    with create_open(out_filepath, mode=mode) as buffer:
        buffer.write(text)


def read_json(in_filepath) -> Dict:
    """
    Read a json-type document into a dict.
    """
    contents = {}
    with open(in_filepath, "r") as buffer:
        contents = json.load(buffer)
    return contents


def save_json(data: Dict, out_filepath: str) -> Dict:
    """
    Save a dictionary into a json file.
    """
    if os.path.splitext(out_filepath)[1].lower() == ".json":
        with create_open(out_filepath, "w") as buffer:
            json.dump(data, buffer, indent=4)
    else:
        raise ValueError("Was expecting a *.json filepath.")


def read_csv(in_filepath: str) -> pd.DataFrame:
    """
    Read a csv file into a dataframe.
    """
    return pd.read_csv(in_filepath)


def save_csv(dataframe: pd.DataFrame, out_filepath: str) -> None:
    """
    Save the dataframe into a CSV file.
    """
    with create_open(out_filepath, "w") as buffer:
        dataframe.to_csv(buffer)


def file_exists(filepath: str, req_ext: str = None) -> bool:
    """
    Given a filepath, determine if the file exists with the required extension.
    """
    if filepath == "":
        return False
    exists = os.path.isfile(os.path.abspath(filepath))
    if req_ext:
        ext_ok = bool( os.path.splitext(filepath)[1] == req_ext )
    else:
        ext_ok = True    
    return bool( exists and ext_ok )


def recursive_find_files(dir: str, template: str) -> List[str]:
    """
    Recursively traverse the subfolders of "dir" to find matching files using a pattern.

    Returns:
        results: a list of paths to folders that match the pattern.
    """
    results = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if file_is_matching(name, template):
                results.append(os.path.join(root, name))
    return results


def find_files_in(root: str, template: Optional[str] = None, abs: bool = True) -> List[str]:
    """
    Find all of the files within the root folder that match the regex template.

    Args:
        root: The path to the root directory.
        template: The template to match files within root. If no template, return all.
        abs: If True, returns the absolute path, otherwise returns the file name only.
    
    Returns:
        matching_files: The files matching the template.
    """
    output = []
    if not os.path.isdir(root):
        return output
    _, folders, files = next(os.walk(root))
    for f in files:
        if abs:
            f = os.path.join(root, f)
        if template is None or file_is_matching(f, template):
            output.append(f)
    return output


def file_is_matching(file: str, template: str) -> bool:
    """
    Checks that the given file is named according to the template.

    The file will be checked against the template. This function will return False if file str
    is not a file.

    Args:
        file: the path (abs, rel, or name) to a file to see if named according to template.
        template: The file template to attempt to match (regex).

    Returns:
        match: True, if the file matches, false otherwise.
    """
    if os.path.isfile(file):
        file = os.path.basename(file)
    else:
        return False  # Not a file.
    if bool( re.match(template, file) ):
        return True
    else:
        return False


def get_file_type(filepath_or_url: str) -> str:
    """
    Get a string with a file type. The returned types are defined according to IANA:
        https://www.iana.org/assignments/media-types/media-types.xhtml

    Args:
        filepath_or_url: the path to a file to check for file type.

    Returns:
        file_type: The type of the file at the path. Can be one of ["application", "audio", 
            "example", "font", "haptics", "image", "message", "model", "multipart", "text", "video"]
            Will return None if unable to identify.
    """
    return mimetypes.guess_type(filepath_or_url)[0]


def copy(src_path: str, dst_path: str):
    """
    General copy method for recursively copying folders or files. The supported copy setups include:
        - folderA -> folderB  # Copy + Rename to folderB
        - fileA -> fileB  # Copy + Rename to fileB
        - fileA -> folder # Copy only, makes new file at "folder/fileA"
    
    Note, this copy function will raise FileExistsError when the dst_path already exists.

    Args:
        src_path: the path to the source file or folder.
        dst_path: the path to the destination file or folder.
    """
    try:
        shutil.copytree(src_path, dst_path)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src_path, dst_path)
        else:
            raise


@contextmanager
def create_open(filepath: str, mode: str = "w", makedirs: bool = True, **kwargs):
    """
    Context manager for creating directories and opening a file.
    """
    # Validate the input.
    ext = os.path.splitext(filepath)[1]
    if not ext:
        raise ValueError("Cannot create_open file without an extension")
    # Create the subdirectories.
    if makedirs:
        parent_folder = os.path.dirname(os.path.abspath(filepath))
        folder.make_dirs(parent_folder, exists_ok=True)
    # Open the file.
    with open(filepath, mode=mode, **kwargs) as buffer:
        yield buffer


