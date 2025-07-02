"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-03-19
"""

import errno
import os
import inspect
import re
import shutil
from typing import List, Dict, Optional

"""
Basic folder utilities. Make, delete, rename, and more with folders.
"""

def make_dirs(dir_path: str, mode: int = 0o777, exists_ok: bool = True) -> None:
    """
    Make a folder.

    Args:
        dir_path: the path to the new folder.
        mode: the permissions for the new folder. "0o" means octal number. Default is 0o777 which
            means "read, write, and execute" permissions for all.
        exists_ok: if True, don't raise error if exists.
    """
    os.makedirs(dir_path, mode=mode, exist_ok=exists_ok)


def remove_dirs(dir_path: str, content_ok: bool = False) -> None:
    """
    Remove a folder.

    Args:
        dir_path: the path to the folder to delete.
        content_ok: if True, delete the files and folders within the dir_path too.
    """
    if content_ok:
        shutil.rmtree(dir_path)
    else:
        os.rmdir(dir_path)


def folder_exists(folderpath: str) -> bool:
    """
    Given a filepath, determine if the file exists.
    """
    if folderpath == "":
        return False
    return os.path.isdir(os.path.abspath(folderpath))


def recursive_find_folders(dir: str, template: str) -> List[str]:
    """
    Recursively travers subfolders of "dir" to find matching folders using a pattern.

    Returns:
        results: a list of paths to folders that match the pattern.
    """
    results = []
    for root, dirs, files in os.walk(dir):
        for name in dirs:
            if folder_is_matching(name, template):
                results.append(os.path.join(root, name))
    return results


def find_folders_in(root: str, template: Optional[str] = None, abs: bool = True) -> List[str]:
    """
    Find all folders within the root folder that matches the template.

    Args:
        root: the path to the root directory.
        template: The template to match folders within root. If no template, return all.
        abs: if true, returns the absolute path. Otherwise, returns the folder name only.

    Returns:
        matching_folders: the folders matching the template.
    """
    output = []
    if not os.path.isdir(root):
        return output
    _, folders, files = next(os.walk(root))
    for f in folders:
        if abs:
            f = os.path.join(root, f)
        if template is None or folder_is_matching(f, template):
            output.append(f)
    return output


def folder_is_matching(folder: str, template: str) -> bool:
    """
    Checks that the given folder is named according to the template. The folder will be checked
    against the template. This function will also return False if 'folder' is not a folder.

    Args:
        folder: the path (abs or rel) to a folder to see if named according to template.
        template: the folder template to attempt to match.

    Returns:
        match: True if the folder matches. False otherwise. Also False if folder does not exist.
    """
    if os.path.isdir(folder):
        folder = os.path.abspath(folder)  # Makes abs path, removes trailing '/'s
        folder = os.path.basename(folder)
    else:
        return False  # Not a folder!
    if bool( re.match(template, folder) ):
        return True
    else:
        return False


def copy_dir(src_dir: str, dst_dir: str, ignore_patterns: List[str] = None):
    """
    General copy method for recursively copying folders or files. The supported copy setups include:
        - folderA -> folderB  # Copy + Rename to folderB
        - fileA -> fileB  # Copy + Rename to fileB
        - fileA -> folder # Copy only, makes new file at "folder/fileA"
    
    Note, this copy function will raise FileExistsError when the dst_path already exists.

    Args:
        src_path: the path to the source file or folder.
        dst_path: the path to the destination file or folder.
        ignore_patterns: types of files or dir patterns to ignore when copying.
            For example -> '*.pyc', 'tmp*'
    """
    try:
        shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(*(ignore_patterns or [])))
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src_dir, dst_dir)
        else:
            raise


def curr_dir() -> str:
    """
    Get the folder that contains the python file containing the calling line.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    return os.path.dirname(module.__file__)