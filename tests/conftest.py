import os
from pathlib import Path
import pytest
from sam2.perceiver.sam2 import SamPerceiver


@pytest.fixture(scope="session", autouse=True, name="model")
def model_setup_fixture():
    """Fixture to setup sam2 perceiver model"""

    cfg_file = Path(
        f"{os.path.abspath(os.path.dirname(__file__))}/test_data/config/test_config.yaml"
    )
    sam_model = SamPerceiver(cfg_file)
    yield sam_model
