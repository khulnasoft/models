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

Startai Models
===========

This repository houses a collection of popular machine learning models written in the `Startai framework <https://github.com/khulnasoft/startai>`_.

Code written in Startai is compatible with PyTorch, TensorFlow, JAX and NumPy.
This means that these models can be integrated into a working pipeline for any of these standard ML frameworks.

The purpose of this repository is to provide reference Startai implementations of common machine learning models, as well as giving a demonstration of how to write custom models in Startai.

Check out our `demos <https://khulnasoft.com/demos/#examples-and-demos>`_ to see these models in action.
In particular, `UNet <https://khulnasoft.com/demos/examples_and_demos/image_segmentation_with_startai_unet.html>`_ 
and `AlexNet <https://khulnasoft.com/demos/examples_and_demos/alexnet_demo.html>`_ demonstrate using models from this repository.

The models can be loaded with pretrained weights, we have tests to ensure that our models give the same output as the reference implementation.
Models can also be initialised with random weights by passing :code:`pretrained=False` to the loading function.

To learn more about Startai, check out `khulnasoft.com <https://khulnasoft.com>`_, our `Docs <https://khulnasoft.com/docs/startai/>`_, and our `GitHub <https://github.com/khulnasoft/startai>`_.

Setting up
------------

.. code-block:: bash

    git clone https://github.com/khulnasoft/models
    cd models
    pip install .

Getting started
-----------------

.. code-block:: python

    import startai
    from startai_models import alexnet
    startai.set_backend("torch")
    model = alexnet()

The pretrained AlexNet model is now ready to be used, and is compatible with any other PyTorch code.
See `this demo <https://khulnasoft.com/demos/examples_and_demos/alexnet_demo.html>`_ for more details.

Navigating this repository
-----------------------------
The models are contained in the startai_models folder.
The functions that automatically load the pretrained weights are found at the end of :code:`model_name.py`, some models have multiple sizes.
The layers are sometimes kept in a separate file, usually named :code:`layers.py`.


**Off-the-shelf models for a variety of domains.**

.. raw:: html

    <div style="display: block;" align="center">
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/supported/empty.png">
    </div>
    <br clear="all" />


**Startai Libraries**

There are a host of derived libraries written in Startai, in the areas of mechanics, 3D vision, robotics, gym environments,
neural memory, pre-trained models + implementations, and builder tools with trainers, data loaders and more. Click on the icons below to learn more!

.. raw:: html

    <div style="display: block;">
        <a href="https://github.com/khulnasoft/mech">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_mech_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_mech.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/vision">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_vision_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_vision.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/robot">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_robot_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_robot.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/gym">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_gym_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_gym.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/startai-mech">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-mech.svg">
        </a>
        <a href="https://pypi.org/project/startai-vision">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-vision.svg">
        </a>
        <a href="https://pypi.org/project/startai-robot">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-robot.svg">
        </a>
        <a href="https://pypi.org/project/startai-gym">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-gym.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/khulnasoft/mech/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"src="https://github.com/khulnasoft/mech/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/khulnasoft/vision/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/vision/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/khulnasoft/robot/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/robot/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/khulnasoft/gym/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/gym/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/khulnasoft/memory">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_memory_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_memory.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/builder">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_builder_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_builder.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/models">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_models_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_models.png">
            </picture>
        </a>
        <a href="https://github.com/khulnasoft/ecosystem">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_ecosystem_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/khulnasoft/khulnasoft.github.io/main/img/externally_linked/logos/startai_ecosystem.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/startai-memory">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-memory.svg">
        </a>
        <a href="https://pypi.org/project/startai-builder">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-builder.svg">
        </a>
        <a href="https://pypi.org/project/startai-models">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/startai-models.svg">
        </a>
        <a href="https://github.com/khulnasoft/ecosystem/actions?query=workflow%3Adocs">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/ecosystem/actions/workflows/docs.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/khulnasoft/memory/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/memory/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/khulnasoft/builder/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/builder/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/khulnasoft/models/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/khulnasoft/models/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />


Citation
--------

::

    @article{lenton2021startai,
      title={Startai: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
