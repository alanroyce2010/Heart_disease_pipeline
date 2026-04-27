Preprocess Module
=================

Overview
--------
Handles data cleaning, missing value imputation, and feature preparation.

Responsibilities
----------------
- Handle missing values
- Identify categorical and numerical columns
- Perform encoding
- Generate derived features

Pipeline Role
-------------
Prepares clean and structured data for feature engineering and modeling.

Input
-----
- Raw dataset (CSV)

Output
------
- Processed dataset (parquet)
- feature_meta.json

Implementation Details
----------------------
.. automodule:: preprocess
   :members:
   :undoc-members:
   :show-inheritance: