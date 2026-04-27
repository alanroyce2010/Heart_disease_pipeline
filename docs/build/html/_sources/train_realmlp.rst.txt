RealMLP Training Module
======================

Overview
--------
Trains a neural network ensemble using RealMLP with cross-validation.

Responsibilities
----------------
- Train multiple models using different hyperparameters
- Perform cross-validation
- Generate ensemble predictions
- Log metrics to MLflow

Model Details
-------------
- Architecture: RealMLP
- Ensemble size: multiple parameter sets
- Evaluation metric: AUC

Pipeline Role
-------------
Core model training stage.

Implementation Details
----------------------
.. automodule:: train_realmlp
   :members:
   :undoc-members:
   :show-inheritance: