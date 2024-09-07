"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2024-03-19
"""

from os.path import join, abspath, basename
from datetime import datetime
from typing import List
from yacs.config import CfgNode

from mlcore.naming import create_name, parse_name, is_name
from mlcore.file import create_open
from mlcore.folder import make_dirs, find_folders_in


"""
A project class to manage IO, file creation, logging, and more related to a new exection.
"""


INPUT_SUBFOLDER = "inputs"
OUTPUT_SUBFOLDER = "outputs"
STANDARD_SUBFOLDERS = [INPUT_SUBFOLDER, OUTPUT_SUBFOLDER]
STANDARD_PROJECT_TAG = "project"

TRAINING_SUBFOLDERS = ["checkpoints"]
LOG_NAME = "stdout.log"
CFG_NAME = "cfg.yaml"


class Project():
    """
    The Project class.
    """
    def __init__(self, project_path: str, cfg: CfgNode):
        """
        Constructor for Project class.

        Args:
            project_path: the path to the project folder.
            cfg: the config node associated with this project.
        """
        self._name = basename(project_path)
        # Get attributes from the project name.
        self._creation_dt, self._uuid, self._type = parse_name(self._name)
        # Save the variables.
        self._path = abspath(project_path)
        self._log_path = join(self._path, LOG_NAME)
        self._log = None
        self._cfg_path = join(self._path, create_name(self._creation_dt, self._uuid, "cfg")+".yaml")
        self._cfg = cfg
        # Make the necessary directories.
        make_dirs(self._path, exists_ok=True)
        self._input_dir = self.make_subfolder(INPUT_SUBFOLDER)
        self._output_dir = self.make_subfolder(OUTPUT_SUBFOLDER)
        
    def _init_validation(self,):
        """
        Validate the class attributes.
        """
        assert is_name(self._name), f"project_path basename must be a name"
    
    @classmethod
    def from_dir(cls, path: str, base_cfg: CfgNode):
        """
        Create a Project by loading from an existing project directory.
        """
        obj = cls(path, base_cfg)
        obj._cfg.merge_from_file(obj._cfg_path)
        return obj

    @classmethod
    def new(cls, root: str, cfg: CfgNode):
        """
        Create a new Project within a root directory.
        """
        name = create_name(tag=STANDARD_PROJECT_TAG)
        path = join(root, name)
        return cls(path, cfg)

    @property
    def path(self) -> str:
        """
        Get the project's folder path.
        """
        return self._path
    
    @property
    def name(self) -> str:
        """
        Get the project's name.
        """
        return self._name

    @property
    def cfg(self) -> CfgNode:
        """
        Return the CfgNode.
        """
        return self._cfg
    
    @property
    def subfolders(self) -> List[str]:
        """
        Return a list of subfolder strings.
        """
        return find_folders_in(self._path, abs=False)
    
    @property
    def subfolder_paths(self) -> List[str]:
        """
        Return the abs paths to the subfolders.
        """
        return find_folders_in(self._path, abs=True)
    
    def make_subfolder(self, foldername: str):
        """
        Make a new subfolder within the project folder.
        """
        folder = join(self._path, foldername)
        make_dirs(folder, exists_ok=True)
        return folder

    def save_cfg(self) -> None:
        """
        Save the project's CfgNode to file.
        """
        with open(self._cfg_path, "w") as stream:
            self._cfg.dump(stream=stream, indent=4)