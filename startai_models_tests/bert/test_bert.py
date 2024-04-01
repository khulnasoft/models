import os
import startai
import pytest
import numpy as np
from startai_models import bert_base_uncased


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_bert(device, fw, batch_shape, load_weights):
    """Test Bert Base Sequence Classification"""

    num_dims = 768
    this_dir = os.path.dirname(os.path.realpath(__file__))

    img_path = os.path.join(this_dir, "bert_inputs.npy")
    inputs = np.load(img_path, allow_pickle=True).tolist()
    model = bert_base_uncased(load_weights)

    inputs = {k: startai.asarray(v) for k, v in inputs.items()}
    logits = model(**inputs)["pooler_output"]
    assert logits.shape == tuple([startai.to_scalar(batch_shape), num_dims])

    if load_weights:
        logits_path = os.path.join(this_dir, "bert_pooled_output.npy")
        ref_logits = np.load(logits_path)
        assert np.allclose(ref_logits, startai.to_numpy(logits), rtol=0.005, atol=0.0005)
