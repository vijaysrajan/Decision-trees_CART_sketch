"""
config_parser module.

Parses YAML/JSON configuration files for theta sketch tree classifier.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


class ConfigParser:
    """
    Parse configuration files (YAML/JSON) for theta sketch tree.

    Supports both YAML (.yaml, .yml) and JSON (.json) formats.

    Expected Format:
    ----------------
    targets:
      positive: "target_yes"
      negative: "target_no"
    hyperparameters:
      criterion: "gini"
      max_depth: 10
      ...
    feature_mapping:
      "age>30": 0       # Column 0 contains binary age>30 values
      "income>50k": 1   # Column 1 contains binary income>50k values
      ...
    """

    ALLOWED_CRITERIA = {"gini", "entropy", "gain_ratio", "binomial", "binomial_chi"}

    def __init__(self):
        """Initialize parser."""
        pass

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Parameters
        ----------
        config_path : str
            Path to configuration file (.yaml, .yml, or .json).

        Returns
        -------
        dict
            Configuration dictionary with keys:
            - 'targets': dict with 'positive' and 'negative'
            - 'hyperparameters': dict of hyperparameters
            - 'feature_mapping': dict of feature mappings

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist.
        ValueError
            If config is invalid or missing required keys.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Detect format from extension
        ext = config_file.suffix.lower()

        if ext in [".yaml", ".yml"]:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        elif ext == ".json":
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {ext}. "
                "Use .yaml, .yml, or .json"
            )

        # Validate configuration
        self.validate_config(config)

        return config

    def parse_feature_mapping(
        self, feature_mapping_config: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Parse feature mapping from config (simple pass-through).

        Parameters
        ----------
        feature_mapping_config : dict
            Feature name → column index mapping

        Returns
        -------
        feature_mapping : dict
            {feature_name: column_index}

        Example
        -------
        Input:
        {
            "age>30": 0,
            "income>50k": 1
        }

        Output: Same as input (already in correct format)
        {
            "age>30": 0,
            "income>50k": 1
        }
        """
        return feature_mapping_config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        # Check required top-level keys
        required_keys = ["targets", "hyperparameters", "feature_mapping"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: '{key}'")

        # Validate targets
        if "positive" not in config["targets"] or "negative" not in config["targets"]:
            raise ValueError("targets must have 'positive' and 'negative' keys")

        # Validate hyperparameters
        if "criterion" in config["hyperparameters"]:
            criterion = config["hyperparameters"]["criterion"]
            if criterion not in self.ALLOWED_CRITERIA:
                raise ValueError(
                    f"Invalid criterion: '{criterion}'. "
                    f"Allowed: {self.ALLOWED_CRITERIA}"
                )

        # Validate feature_mapping
        if not isinstance(config["feature_mapping"], dict):
            raise ValueError("feature_mapping must be a dictionary")

        for feature_name, column_idx in config["feature_mapping"].items():
            if not isinstance(feature_name, str):
                raise ValueError(
                    f"Feature names must be strings. Got: {type(feature_name)}"
                )
            if not isinstance(column_idx, int) or column_idx < 0:
                raise ValueError(
                    f"Column indices must be non-negative integers. "
                    f"Got {column_idx} for feature '{feature_name}'"
                )
