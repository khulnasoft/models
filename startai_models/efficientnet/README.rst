.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
   :width: 100%
   :class: only-dark


.. raw:: html

    <br/>
    <a href="https://pypi.org/project/startai-models">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/startai-models.svg">
    </a>
    <a href="https://github.com/khulnasoft/models/actions?query=workflow%3Adocs">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/khulnasoft/models/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/khulnasoft/models/actions?query=workflow%3Anightly-tests">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/khulnasoft/models/actions/workflows/nightly-tests.yml/badge.svg">
    </a>
    <a href="https://discord.gg/G4aR9Q7DTN">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

EfficientNet
===========

`EfficientNet <https://arxiv.org/abs/1905.11946>`_ represents a convolutional neural network structure and scaling approach that consistently adjusts depth, 
width, and resolution using a combined coefficient. Diverging from traditional methods that unevenly modify these aspects, 
the EfficientNet approach maintains uniform adjustments to network dimensions by utilizing a predefined set of scaling coefficients. 

This method employs a compound coefficient to ensure consistent modifications in network attributes, resulting in a more systematic approach.
The rationale behind the compound scaling method is intuitive: as the input image size increases, 
the network requires additional layers to enhance its receptive field, along with increased channels to capture intricate patterns specific to larger images.

Getting started
-----------------

.. code-block:: python

    import startai
    from startai_models.efficientnet import efficientnet_b0
    startai.set_backend("torch")

    # Instantiate efficientnet_b0 model
    startai_efficientnet_b0 = efficientnet_b0(pretrained=True)

    # Convert the Torch image tensor to an Startai tensor and adjust dimensions
    img = startai.asarray(torch_img.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")

    # Compile the Startai efficientnet_b0 model with the Startai image tensor
    startai_efficientnet_b0.compile(args=(img,))

    # Pass the Startai image tensor through the Startai efficientnet_b0 model and apply softmax
    output = startai.softmax(startai_efficientnet_b0(img))

    # Get the indices of the top 3 classes from the output probabilities
    classes = startai.argsort(output[0], descending=True)[:3] 

    # Retrieve the logits corresponding to the top 3 classes
    logits = startai.gather(output[0], classes) 

    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
    print("Categories of the top 3 classes are:", [categories[i] for i in classes.to_list()])

    `Indices of the top 3 classes are: startai.array([282, 281, 285], dev=gpu:0)`
    `Logits of the top 3 classes are: startai.array([0.60317987, 0.18620452, 0.07535177], dev=gpu:0)`
    `Categories of the top 3 classes are: ['tiger cat', 'tabby', 'Egyptian cat']`

Citation
--------

::

    @article{
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
      author={Mingxing Tan and Quoc V. Le},
      journal={arXiv preprint arXiv:1905.11946},
      year={2020}
    }


    @article{lenton2021startai,
      title={Startai: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
