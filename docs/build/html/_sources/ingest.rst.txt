Ingest Module
=============

Overview
--------
The ingest module is responsible for loading raw datasets, validating them,
encoding the target variable, and saving the processed dataset in parquet format.

This is the first stage of the pipeline.

Responsibilities
----------------
- Load training dataset from CSV
- Encode target labels (Presence → 1, Absence → 0)
- Validate dataset integrity
- Save processed dataset to parquet format

Pipeline Role
-------------
This module acts as the entry point of the data pipeline and prepares data
for downstream feature engineering and training stages.

Input
-----
- data/raw/train.csv

Output
------
- data/processed/train.parquet

Implementation Details
----------------------
.. automodule:: ingest
   :members:
   :undoc-members:
   :show-inheritance: