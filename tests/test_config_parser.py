"""
Tests for config_parser.

Tests YAML and JSON configuration loading and validation.
"""

import pytest
import yaml
import json
from pathlib import Path
from theta_sketch_tree.config_parser import ConfigParser


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "targets": {"positive": "target_yes", "negative": "target_no"},
        "hyperparameters": {"criterion": "gini", "max_depth": 10, "min_samples_split": 20},
        "feature_mapping": {"age>30": 0, "income>50k": 1, "city=NY": 2},
    }


@pytest.fixture
def yaml_config_file(tmp_path, sample_config):
    """Create a YAML config file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return str(config_file)


@pytest.fixture
def json_config_file(tmp_path, sample_config):
    """Create a JSON config file."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(sample_config, f, indent=2)
    return str(config_file)


class TestConfigParser:
    """Test suite for ConfigParser."""

    def test_load_yaml_config(self, yaml_config_file):
        """Test loading a YAML configuration file."""
        parser = ConfigParser()
        config = parser.load(yaml_config_file)

        assert "targets" in config
        assert "hyperparameters" in config
        assert "feature_mapping" in config
        assert config["targets"]["positive"] == "target_yes"
        assert config["targets"]["negative"] == "target_no"
        assert config["hyperparameters"]["criterion"] == "gini"
        assert config["feature_mapping"]["age>30"] == 0

    def test_load_json_config(self, json_config_file):
        """Test loading a JSON configuration file."""
        parser = ConfigParser()
        config = parser.load(json_config_file)

        assert "targets" in config
        assert "hyperparameters" in config
        assert "feature_mapping" in config
        assert config["targets"]["positive"] == "target_yes"
        assert config["hyperparameters"]["criterion"] == "gini"

    def test_parse_feature_mapping(self):
        """Test parsing feature mapping (pass-through)."""
        parser = ConfigParser()
        feature_mapping = {"age>30": 0, "income>50k": 1}
        result = parser.parse_feature_mapping(feature_mapping)

        assert result == feature_mapping

    def test_validate_config_valid(self, sample_config):
        """Test validation with valid config."""
        parser = ConfigParser()
        # Should not raise any exception
        parser.validate_config(sample_config)

    def test_validate_config_missing_targets(self, sample_config):
        """Test validation error when targets key is missing."""
        parser = ConfigParser()
        del sample_config["targets"]

        with pytest.raises(ValueError, match="Missing required config key: 'targets'"):
            parser.validate_config(sample_config)

    def test_validate_config_missing_hyperparameters(self, sample_config):
        """Test validation error when hyperparameters key is missing."""
        parser = ConfigParser()
        del sample_config["hyperparameters"]

        with pytest.raises(
            ValueError, match="Missing required config key: 'hyperparameters'"
        ):
            parser.validate_config(sample_config)

    def test_validate_config_missing_feature_mapping(self, sample_config):
        """Test validation error when feature_mapping key is missing."""
        parser = ConfigParser()
        del sample_config["feature_mapping"]

        with pytest.raises(
            ValueError, match="Missing required config key: 'feature_mapping'"
        ):
            parser.validate_config(sample_config)

    def test_validate_config_missing_positive_target(self, sample_config):
        """Test validation error when positive target is missing."""
        parser = ConfigParser()
        del sample_config["targets"]["positive"]

        with pytest.raises(
            ValueError, match="targets must have 'positive' and 'negative' keys"
        ):
            parser.validate_config(sample_config)

    def test_validate_config_invalid_criterion(self, sample_config):
        """Test validation error with invalid criterion."""
        parser = ConfigParser()
        sample_config["hyperparameters"]["criterion"] = "invalid_criterion"

        with pytest.raises(ValueError, match="Invalid criterion"):
            parser.validate_config(sample_config)

    def test_validate_config_non_dict_feature_mapping(self, sample_config):
        """Test validation error when feature_mapping is not a dict."""
        parser = ConfigParser()
        sample_config["feature_mapping"] = ["age>30", "income>50k"]

        with pytest.raises(ValueError, match="feature_mapping must be a dictionary"):
            parser.validate_config(sample_config)

    def test_validate_config_non_string_feature_name(self, sample_config):
        """Test validation error when feature name is not a string."""
        parser = ConfigParser()
        sample_config["feature_mapping"][123] = 0

        with pytest.raises(ValueError, match="Feature names must be strings"):
            parser.validate_config(sample_config)

    def test_validate_config_negative_column_index(self, sample_config):
        """Test validation error when column index is negative."""
        parser = ConfigParser()
        sample_config["feature_mapping"]["age>30"] = -1

        with pytest.raises(
            ValueError, match="Column indices must be non-negative integers"
        ):
            parser.validate_config(sample_config)

    def test_validate_config_non_integer_column_index(self, sample_config):
        """Test validation error when column index is not an integer."""
        parser = ConfigParser()
        sample_config["feature_mapping"]["age>30"] = "0"

        with pytest.raises(
            ValueError, match="Column indices must be non-negative integers"
        ):
            parser.validate_config(sample_config)

    def test_load_file_not_found(self):
        """Test error when config file doesn't exist."""
        parser = ConfigParser()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            parser.load("/nonexistent/config.yaml")

    def test_load_unsupported_format(self, tmp_path):
        """Test error when config file has unsupported format."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("invalid format")

        parser = ConfigParser()

        with pytest.raises(ValueError, match="Unsupported config format"):
            parser.load(str(config_file))

    def test_load_yml_extension(self, tmp_path, sample_config):
        """Test loading YAML file with .yml extension."""
        config_file = tmp_path / "config.yml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        parser = ConfigParser()
        config = parser.load(str(config_file))

        assert config["targets"]["positive"] == "target_yes"
