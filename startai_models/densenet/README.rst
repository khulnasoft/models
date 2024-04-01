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

DenseNet
===========

`DenseNets <https://arxiv.org/abs/1608.06993>`_ are a powerful and efficient CNN architecture that can be used for a variety of image classification tasks. 
They are especially suitable for tasks where efficiency and robustness to overfitting are important.

DenseNets are based on a novel connectivity pattern called "dense connections". In a dense network, each layer is connected to every other layer in the network.
This allows the network to learn long-range dependencies in the input images, which is important for image classification tasks.
In addition to dense connections, DenseNets also use a number of other techniques to improve their efficiency and accuracy. 
These techniques include:Transition layers, Bottleneck layers and Dropout


Getting started
-----------------

.. code-block:: python

    import startai
    from startai_models.densenet import densenet121
    startai.set_backend("torch")

    # Instantiate densenet121 model
    startai_densenet121 = densenet121(pretrained=True)

    # Convert the Torch image tensor to an Startai tensor and adjust dimensions
    img = startai.asarray(startai_densenet121.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")

    # Compile the Startai densenet121 model with the Startai image tensor
    startai_densenet121.compile(args=(img,))

    # Pass the Startai image tensor through the Startai densenet121 model and apply softmax
    output = startai.softmax(startai_densenet121(img))

    # Get the indices of the top 3 classes from the output probabilities
    classes = startai.argsort(output[0], descending=True)[:3] 

    # Retrieve the logits corresponding to the top 3 classes
    logits = startai.gather(output[0], classes) 


Citation
--------

::

    @article{
      title={Densely Connected Convolutional Networks},
      author={Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Q. Weinberger},
      journal={arXiv preprint arXiv:1608.06993,
      year={2018}
    }


    @article{lenton2021startai,
      title={Startai: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
