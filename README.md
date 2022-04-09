# Tensorflow Pipeline

An framework built to be a component in a pipeline of data processing using `Tensorflow`.

This framework is used to:
* Read labelled data from upstream data providers.
* Preprocess data and split them into a training set and a validation set.
* Load model configurations, train the model and save the model to filesystem for future use.
* Evaluate the model with the validation set and save the visualization of diagnostics.
* Provide an interface for downstream data users to call the model for prediction.

Note that this repo is a pipeline framework only and the model construction
itself is **not** included: models will be dynamically loaded to the pipeline
according to the path defined in the `config.json` file.