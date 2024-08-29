from pathlib import Path
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Any
from perceiver.defs import PerceiverOutput
from perceiver.perceiver_factory import PerceiverFactory
from perceiver.perceiver_base import PerceiverBase
from enum import unique, Enum


@unique
class PromptType(Enum):
    """
    Enumerated type for defining prompt type
    TODO Define in a defs file somewhere
    """

    BOX = 1
    POINTS = 2
    MASK = 3


@PerceiverFactory.register("sam2")
class SamPerceiver(PerceiverBase):
    """
    SAM 2 perceiver wrapper
    """

    def __init__(self, config_file: Path):
        """
        Builds SAM2 model using config parameters provided

        Args:
            config_file (Path): Full path to yaml config file
        """

        super().__init__(config_file)
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

    @staticmethod
    def _check_inputs(input_dict: dict[str, Any]) -> None:
        """Checks if inputs are in correct formats"""
        assert "IMAGE" in input_dict, "input dict must contain IMAGE"
        assert "PROMPT" in input_dict, "input dict must contain PROMPT"
        assert "PROMPT_TYPE" in input_dict, "input dict must contain PROMPT_TYPE"
        assert isinstance(
            input_dict["PROMPT_TYPE"], PromptType
        ), "must be of type PromptType"
        assert isinstance(
            input_dict["IMAGE"], np.ndarray
        ), "input image must be numpy array"
        assert isinstance(
            input_dict["PROMPT"], np.ndarray
        ), "prompt must be numpy array"

        # Check format of prompt
        match input_dict["PROMPT_TYPE"]:
            case PromptType.BOX:
                assert input_dict["PROMPT"].ndim == 2, "Bounding box prompt must be 2D"
                assert (
                    input_dict["PROMPT"].shape[1] == 4
                ), "Bounding box prompt must be of shape N X 4"
            case PromptType.MASK:
                assert (
                    input_dict["PROMPT"].shape == input_dict["IMAGE"].shape[:2]
                ), "Mask prompt shape does not match image shape"
            case PromptType.POINTS:
                assert input_dict["PROMPT"].ndim == 2, "Points prompt must be 2D"
                assert (
                    input_dict["PROMPT"].shape[1] == 3
                ), "Points prompt must be of shape N X 3."

    def __call__(self, input_dict: dict[str, Any]) -> PerceiverOutput:
        """

        Args:
            input_dict:

        Returns:

        """

        # Check inputs
        try:
            self._check_inputs(input_dict)
        except AssertionError as e:
            raise AssertionError(f"SAM 2 error: {e}")

        self.predictor.set_image(input_dict["IMAGE"])
