Featurize Module
================

Overview
--------
Generates advanced features to improve model performance.

Feature Types
-------------
- Digit-based features
- Binning features (quantile, uniform, rounding)
- Categorical transformations

Responsibilities
----------------
- Create new feature representations
- Ensure consistency across datasets
- Maintain feature metadata

Pipeline Role
-------------
Enhances raw features into model-ready inputs.

Input
-----
- data/processed/train.parquet

Output
------
- train_feat.parquet
- feature_meta.json

Implementation Details
----------------------
.. automodule:: featurize
   :members:
   :undoc-members:
   :show-inheritance: