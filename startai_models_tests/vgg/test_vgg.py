import os
import numpy as np
import pytest
import random
import startai
from startai_models_tests import helpers
from startai_models.vgg import (
    vgg11,
    vgg13,
    vgg16,
)


VARIANTS = {
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
}

PREDS = {
    "vgg11": [282, 281, 285],
    "vgg13": [281, 282, 292],
    "vgg16": [282, 281, 292],
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = startai.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_vgg_img_classification(device, fw, data_format):
    """Test VGG image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = startai.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
            224,
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
        true_indices = np.sort(np.array(PREDS[model_var]))
        calc_indices = np.sort(np.argsort(np_out)[-3:][::-1])
        assert np.array_equal(true_indices, calc_indices)
