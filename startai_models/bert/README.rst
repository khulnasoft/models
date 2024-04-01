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

BERT
===========

`BERT <https://arxiv.org/abs/1810.04805>`_ short for Bidirectional Encoder Representations from Transformers, differentiates itself from 
recent language representation models by its focus on pretraining deep bidirectional representations from unannotated text. 
This approach involves considering both left and right context in all layers.
Consequently, the pretrained BERT model can be enhanced with just a single additional output layer to excel in various tasks, 
such as question answering and language inference. This achievement is possible without extensive modifications to task-specific architecture.

Getting started
-----------------

.. code-block:: python

    import startai
    startai.set_backend("torch")

    # Instantiate Bert
    startai_bert = startai_models.bert_base_uncased(pretrained=True)

    # Convert the input data to Startai tensors
    startai_inputs = {k: startai.asarray(v.numpy()) for k, v in inputs.items()}

    # Compile the Startai BERT model with the Startai input tensors
    startai_bert.compile(kwargs=startai_inputs)

    # Pass the Startai input tensors through the Startai BERT model and obtain the pooler output
    startai_output = startai_bert(**startai_inputs)['pooler_output']


See `this demo <https://github.com/khulnasoft/demos/blob/main/examples_and_demos/bert_demo.ipynb>`_ for more usage example.

Citation
--------

::

    @article{
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
      author={Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova},
      journal={arXiv preprint arXiv:1810.04805},
      year={2019}
    }


    @article{lenton2021startai,
      title={Startai: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
