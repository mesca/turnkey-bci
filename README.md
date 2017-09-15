# Towards Turnkey Brain-Computer Interfaces

## Overview

An electroencephalogram (EEG) is a recording of the electrical activity of the brain using electrodes placed on the scalp. Signals from EEGs can be interpreted in the frequency domain (neural oscillations) or in the time domain (event-related potentials or ERPs). These signals are highly subject-specific, depend on experimental conditions and vary between recording sessions. For this reason, most Brain-Computer interfaces (BCIs) require a calibration phase, during which data is collected in order to train a subject-specific classifier. This calibration phase is often perceived as an hindrance, and can be difficult to complete for subjects with a limited attention span.

In this study, we focused on the problem of transfer learning in ERP-based BCIs. Specifically, we attempted to build a generic LSTM model that works reasonably well across different subjects and attention tasks.

## Getting started

* Downloaded the dataset from: [http://dx.doi.org/10.14279/depositonce-5523](http://dx.doi.org/10.14279/depositonce-5523). Only the ``EEG.zip`` file is required for this project.
* Unzip the archive and move ``EEG.h5`` to the ``data/input/`` directory.
* Explore with the provided notebook. Pre-computed models are provided.
* Tweak the code, optimize, and train new models!

The dataset originates from a study that can be found at [https://doi.org/10.1371/journal.pone.0165556](https://doi.org/10.1371/journal.pone.0165556).

## Dependencies

These are required for running the project:

```
$ conda create --name mne python=2
$ source activate mne
$ conda install scipy matplotlib seaborn scikit-learn mayavi ipython-notebook pytables h5py
$ pip install mne --upgrade
$ conda install -c conda-forge keras tensorflow
```

## Files

* ``report.pdf``: the project report.
* ``notebook.html``: the exported notebook, with all cells ran.
* ``src/notebook.ipynb``: the project notebook, with all cells ran.
* ``src/erp.py``: the main code.
* ``src/train.py``: we do not recommend to build the models from the notebook, as it can take a very long time ; instead, run this file from the command line, and optionaly set the ``implementation`` variable to ``2`` to use the GPU.
* ``src/lstm_graph.py``: a small script to generate Keras visualization.
* ``data/input``: the input directory, where ``EEG.h5`` must be moved.
* ``data/output``: the output directory, where pre-computed results are located.
