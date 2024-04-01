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

AlexNet
===========

`AlexNet <https://arxiv.org/abs/1404.5997>`_ competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. 

The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. 
The original paper’s primary result was that the depth of the model was essential for its high performance, which was computationally expensive, 
but made feasible due to the utilization of graphics processing units (GPUs) during training.

Getting started
-----------------

.. code-block:: python

    import startai
    startai.set_backend("torch")
    from startai_models.alexnet import alexnet

    # Instantiate the AlexNet Model
    startai_alexnet = alexnet()

    # Complile the model with the image preprocessed using torch
    startai_alexnet = startai.compile(startai_alexnet, args=(startai.asarray(torch_img.cuda()),))

    # Pass the processed image to the model
    output = startai.softmax(startai_alexnet(startai.asarray(img))) 
    classes = startai.argsort(output[0], descending=True)[:3]  # get the top 3 classes
    logits = startai.gather(output[0], classes)  # get the logits

    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
    print("Categories of the top 3 classes are:", [categories[i] for i in classes.to_list()])


    `Indices of the top 3 classes are: startai.array([282, 281, 285], dev=gpu:0)`
    `Logits of the top 3 classes are: startai.array([0.64773697, 0.29496649, 0.04526037], dev=gpu:0)`
    `Categories of the top 3 classes are: ['tiger cat', 'tabby', 'Egyptian cat']`


The pretrained AlexNet model is now ready to be used, and is compatible with any Tensorflow, Jax and PyTorch code.
See `this demo <https://khulnasoft.com/demos/examples_and_demos/alexnet_demo.html>`_ for more usage example.

Citation
--------

::

    @article{
      title={One weird trick for parallelizing convolutional neural networks},
      author={Alex Krizhevsky},
      journal={arXiv preprint arXiv:1404.5997},
      year={2014}
    }


    @article{lenton2021startai,
      title={Startai: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
