System Architecture
===================

Overview
--------
The Heart Disease MLOps system is designed as a modular, loosely coupled architecture
where each component operates independently and communicates via well-defined interfaces.

The system follows a typical MLOps lifecycle:
data ingestion → preprocessing → training → serving → monitoring.

Architecture Diagram
--------------------

.. code-block:: text

        ┌───────────────┐
        │   User (UI)   │
        │  Streamlit    │
        └──────┬────────┘
               │ REST API
               ▼
        ┌───────────────┐
        │ MLflow Serve  │
        │ Model API     │
        └──────┬────────┘
               │
               ▼
        ┌───────────────┐
        │  Model Store  │
        │   (MLflow)    │
        └──────┬────────┘
               │
   ┌───────────▼───────────┐
   │   Training Pipeline    │
   │  (DVC + Python)       │
   └───────────┬───────────┘
               │
        ┌──────▼───────┐
        │  Raw Data    │
        └──────────────┘


Components
----------

1. Frontend (Streamlit)
~~~~~~~~~~~~~~~~~~~~~~~
- Provides user interface for predictions
- Displays monitoring dashboards
- Sends requests to backend via REST API

2. Model Serving (MLflow)
~~~~~~~~~~~~~~~~~~~~~~~~~
- Hosts trained models
- Exposes prediction endpoint ``/invocations``
- Handles inference requests

3. Model Registry (MLflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Stores trained models
- Manages model versions
- Promotes best model to production

4. Training Pipeline (DVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Handles data preprocessing
- Performs feature engineering
- Trains multiple models

5. Orchestration (Airflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Detects new data
- Triggers pipeline execution
- Manages workflow dependencies

6. Monitoring System
~~~~~~~~~~~~~~~~~~~~
- Prometheus collects metrics
- Grafana visualizes metrics
- Tracks:
  - Prediction counts
  - Drift scores
  - Latency

Design Principles
----------------

Loose Coupling
~~~~~~~~~~~~~~
Frontend and backend are independent systems connected via REST API.

Modularity
~~~~~~~~~~
Each pipeline stage is isolated and reusable.

Scalability
~~~~~~~~~~~
Components can be scaled independently (e.g., model serving, UI).

Reliability
~~~~~~~~~~~
Airflow ensures pipeline automation and failure handling.