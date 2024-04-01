import os
import numpy as np
import pytest
import random
import startai
from startai_models_tests import helpers
from startai_models.vit import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
)


VARIANTS = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
}


LOGITS = {
    "vit_b_16": [282, 281, 285, 287, 292],
    "vit_b_32": [282, 281, 285, 287, 292],
    "vit_l_16": [255, 281, 282, 285, 292],
    "vit_l_32": [282, 281, 285, 287, 292],
}


load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = startai.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_vit_img_classification(device, fw, data_format):
    """Test ViT image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = startai.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
            256,
            224,
            data_format=data_format,
            to_startai=True,
        ),
    )

    # Create model
    model.v = startai.asarray(v)
    logits = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([startai.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = startai.to_numpy(logits[0])
        true_indices = np.sort(np.array(LOGITS[model_var]))
        calc_indices = np.sort(np.argsort(np_out)[-5:][::-1])
        assert np.array_equal(true_indices, calc_indices)
