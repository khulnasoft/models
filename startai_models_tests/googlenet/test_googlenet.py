import os
import random
import startai
import pytest
import numpy as np

from startai_models.googlenet import inceptionNet_v1
from startai_models_tests import helpers


load_weights = random.choice([False, True])
model = inceptionNet_v1(pretrained=load_weights)
v = startai.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_GoogleNet_tiny_img_classification(device, fw, data_format):
    """Test GoogleNet image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
        256,
        224,
        data_format=data_format,
        to_startai=True,
    )

    model.v = startai.asarray(v)
    logits, _, _ = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([startai.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = startai.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285])
        calc_indices = np.argsort(np_out)[-3:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        # true_logits = np.array([0.2539, 0.2391, 0.1189])
        # calc_logits = np.take(np_out, calc_indices)
        # assert np.allclose(true_logits, calc_logits, rtol=1e-1)
