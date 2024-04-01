import os
import startai
import pytest
import random
import numpy as np
from startai_models_tests import helpers
from startai_models.efficientnet import (
    efficientnet_b0,
)

VARIANTS = {
    "efficientnet_b0": efficientnet_b0,
    # "efficientnet_b1": efficientnet_b1,
    # "efficientnet_b2": efficientnet_b2,
    # "efficientnet_b3": efficientnet_b3,
    # "efficientnet_b4": efficientnet_b4,
    # "efficientnet_b5": efficientnet_b5,
    # "efficientnet_b6": efficientnet_b6,
    # "efficientnet_b7": efficientnet_b7,
    # "efficientnet_v2_l": efficientnet_v2_l,
    # "efficientnet_v2_m": efficientnet_v2_m,
    # "efficientnet_v2_s": efficientnet_v2_s,
}


load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = startai.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_efficientnet_b0_img_classification(device, fw, data_format):
    """Test EfficientNet image classification."""
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
        true_indices = np.array([282, 281, 285])
        calc_indices = np.argsort(np_out)[-3:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        # true_logits = np.array([8.69796944, 7.64586592, 6.92855835])
        # calc_logits = np.take(np_out, calc_indices)
        # assert np.allclose(true_logits, calc_logits, rtol=1e-1)
