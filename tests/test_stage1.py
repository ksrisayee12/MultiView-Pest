"""
Stage 1 Unit Tests for Multi-View Agricultural Analysis System.
Uses Python built-in unittest framework (0 external test dependencies required).
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.validate_datasets import (
    CONFIGS_DIR,
    DATA_DIR,
    check_configurations,
    check_directories,
    load_yaml_simple,
    run_validation_report,
)


class TestStage1Foundation(unittest.TestCase):
    """Unit tests for Stage 1 baseline files, directory structure, and configuration integrity."""

    def test_required_directories_exist(self):
        """Verify all mandated project directories exist on disk."""
        dir_results = check_directories()
        for dir_path, exists in dir_results.items():
            self.assertTrue(exists, f"Required project directory is missing: {dir_path}")

    def test_configuration_files_exist_and_readable(self):
        """Verify configuration files are present and readable."""
        config_results = check_configurations()
        self.assertTrue(len(config_results) > 0, "No configuration files checked.")
        for cfg_name, res in config_results.items():
            self.assertNotEqual(
                res["status"],
                "MISSING",
                f"Configuration file {cfg_name} is missing.",
            )

    def test_disease_class_mappings_present(self):
        """Verify PlantVillage official tomato labels are explicitly mapped to project labels."""
        disease_cfg_path = CONFIGS_DIR / "disease_classes.yaml"
        self.assertTrue(disease_cfg_path.exists(), "disease_classes.yaml is missing")

        with open(disease_cfg_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_official_labels = [
            "Tomato___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        ]

        expected_project_labels = [
            "healthy",
            "bacterial_spot",
            "early_blight",
            "late_blight",
            "tomato_yellow_leaf_curl_virus",
        ]

        for official_label in expected_official_labels:
            self.assertIn(
                official_label,
                content,
                f"Missing official PlantVillage label: {official_label}",
            )

        for project_label in expected_project_labels:
            self.assertIn(
                project_label,
                content,
                f"Missing project class label: {project_label}",
            )

    def test_no_fake_drone_dataset_configured_as_verified(self):
        """Ensure no unverified UAV drone dataset is falsely marked as VERIFIED."""
        sources_cfg_path = CONFIGS_DIR / "dataset_sources.yaml"
        self.assertTrue(sources_cfg_path.exists(), "dataset_sources.yaml is missing")

        with open(sources_cfg_path, "r", encoding="utf-8") as f:
            content = f.read()

        # If drone aerial entry exists, ensure its status is not falsely set to VERIFIED without proof
        if "drone_aerial:" in content:
            self.assertNotIn(
                'status: "VERIFIED"',
                content.split("drone_aerial:")[1],
                "Drone dataset must not be marked as VERIFIED before actual dataset selection",
            )

    def test_dataset_validator_execution(self):
        """Test that the dataset validator report function executes cleanly."""
        validation_success = run_validation_report()
        self.assertTrue(validation_success, "Dataset validator reported failure.")


if __name__ == "__main__":
    unittest.main()
