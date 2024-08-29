from sam2.perceiver.sam2 import PromptType

import os
import pytest
import cv2
import torch
import numpy as np

DATA_DIR = f"{os.path.abspath(os.path.dirname(__file__))}/test_data/"
TEST_IMAGE = f"{DATA_DIR}/cat_dog.jpeg"


@pytest.fixture(name="input_image")
def image_fixture():
    """Fixture to load image"""
    image = cv2.imread(TEST_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yield image


def test_no_image(model):
    """Tests for assertion error when no image provided"""
    no_image = {
        "PROMPT": np.array([[1, 2, 3, 4]], dtype=np.uint32),
        "PROMPT_TYPE": PromptType.BOX,
    }

    with pytest.raises(
        AssertionError, match="SAM 2 error: input dict must contain IMAGE"
    ):
        model(no_image)


def test_no_prompt(model, input_image):
    """Tests for assertion error when no prompt provided"""
    no_prompt = {"IMAGE": input_image, "PROMPT_TYPE": PromptType.BOX}

    with pytest.raises(
        AssertionError, match="SAM 2 error: input dict must contain PROMPT"
    ):
        model(no_prompt)


def test_no_prompt_type(model, input_image):
    no_type = {
        "PROMPT": np.array([[1, 2, 3, 4]], dtype=np.uint32),
        "IMAGE": torch.from_numpy(input_image),
    }

    with pytest.raises(
        AssertionError, match="SAM 2 error: input dict must contain PROMPT_TYPE"
    ):
        model(no_type)


def test_invalid_image(model, input_image):
    """Tests for assertion error when invalid image provided"""
    invalid_image = {
        "PROMPT": np.array([[1, 2, 3, 4]], dtype=np.uint32),
        "PROMPT_TYPE": PromptType.BOX,
        "IMAGE": torch.from_numpy(input_image),
    }

    with pytest.raises(
        AssertionError, match="SAM 2 error: input image must be numpy array"
    ):
        model(invalid_image)


def test_invalid_prompts(model, input_image):
    """Tests for assertion error when invalid prompts provided"""

    # BOX
    invalid_prompt = {
        "IMAGE": input_image,
        "PROMPT_TYPE": PromptType.BOX,
        "PROMPT": np.array([[1, 2, 3]], dtype=np.int32),
    }  # BBOX needs to be Nx4

    with pytest.raises(
        AssertionError, match="SAM 2 error: Bounding box prompt must be of shape N X 4"
    ):
        model(invalid_prompt)

    invalid_prompt["PROMPT"] = np.array(
        [1, 2, 3, 4], dtype=np.int32
    )  # Box should be 2D
    with pytest.raises(
        AssertionError, match="SAM 2 error: Bounding box prompt must be 2D"
    ):
        model(invalid_prompt)

    # MASK
    mask = np.ones(shape=(input_image.shape[0], input_image.shape[0]))
    invalid_prompt["PROMPT_TYPE"] = PromptType.MASK
    invalid_prompt["PROMPT"] = mask
    with pytest.raises(
        AssertionError,
        match="SAM 2 error: Mask prompt shape does not match image shape",
    ):
        model(invalid_prompt)

    # POINTS
    invalid_prompt["PROMPT"] = np.array(
        [[1, 2]], dtype=np.int32
    )  # Points needs to be Nx3
    invalid_prompt["PROMPT_TYPE"] = PromptType.POINTS

    with pytest.raises(
        AssertionError, match="SAM 2 error: Points prompt must be of shape N X 3"
    ):
        model(invalid_prompt)

    invalid_prompt["PROMPT"] = np.array(
        [1, 2, 3], dtype=np.int32
    )  # Points shoull be 2D
    with pytest.raises(AssertionError, match="SAM 2 error: Points prompt must be 2D"):
        model(invalid_prompt)
