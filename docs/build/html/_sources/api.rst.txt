API Documentation
=================

Overview
--------
The system exposes a REST API for model inference using MLflow serving.

All communication between the frontend and backend happens through this API.

Base URL
--------
::

   http://localhost:8001

Endpoint
--------

POST /invocations
~~~~~~~~~~~~~~~~~

Description
-----------
Predicts the probability of heart disease for a given patient.

Request Format
--------------

The API expects input in ``dataframe_split`` format.

Example:

.. code-block:: json

   {
     "dataframe_split": {
       "columns": [
         "Age", "BP", "Cholesterol", "Max HR", "ST depression",
         "Sex", "Chest pain type", "FBS over 120", "EKG results",
         "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"
       ],
       "data": [
         [55, 130, 230, 150, 1.0, 1, 3, 0, 1, 0, 2, 0, 3]
       ]
     }
   }

Response
--------

.. code-block:: json

   {
     "predictions": [0.73]
   }

Response Fields
--------------

- ``predictions`` : list of float  
  Probability of heart disease (0 to 1)

Error Handling
--------------

- 400 Bad Request
  Invalid input format

- 500 Internal Server Error
  Model inference failed

- Timeout
  Backend service not reachable

Usage Flow
----------

1. User inputs data in the Streamlit UI
2. UI sends POST request to API
3. MLflow processes input and generates prediction
4. Response is returned and displayed

Design Considerations
---------------------

Stateless API
~~~~~~~~~~~~~
Each request is independent and does not rely on previous calls.

Loose Coupling
~~~~~~~~~~~~~~
Frontend does not depend on model implementation.

Scalability
~~~~~~~~~~~
API can be deployed independently and scaled horizontally.