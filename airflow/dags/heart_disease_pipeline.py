"""
Airflow DAG: heart_disease_pipeline
────────────────────────────────────
Watches /opt/airflow/data/ for a new train.csv.
When found, stages it to the shared volume, runs the full DVC pipeline
using ephemeral Docker containers, and restarts the monitoring services.

Email notifications:
  - Each stage sends a failure email if it errors
  - A final success email is sent when the full pipeline completes
  - A final failure email is sent if any critical stage fails
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
from docker.types import DeviceRequest, Mount

# ── Constants ──────────────────────────────────────────────────────────────
PIPELINE_DIR = os.getenv("PIPELINE_DIR",          "/opt/airflow/pipeline")
TRAINING_DIR = os.getenv("TRAINING_DIR",          "/opt/airflow/data")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI",   "http://mlflow:5000")
MODEL_NAME   = os.getenv("MLFLOW_MODEL_NAME",     "heart_disease_best")
ALERT_EMAIL  = os.getenv("PIPELINE_ALERT_EMAIL",  "alanroyce2010@gmail.com")

DEFAULT_ARGS = {
    "owner":            "mlops",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,   # we handle emails manually below
    "email_on_retry":   False,
}

DVC_MOUNTS = [
    Mount(source="/home/alan/realmlp_pipeline/pipeline",  target="/opt/pipeline", type="bind"),
    Mount(source="/home/alan/realmlp_pipeline/mlflow_data", target="/mlflow",     type="bind"),
]
ENV_VARS       = {"MLFLOW_TRACKING_URI": MLFLOW_URI, "MLFLOW_MODEL_NAME": MODEL_NAME}
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "realmlp_pipeline_hdnet")
GPU_REQUEST    = [DeviceRequest(count=-1, capabilities=[["gpu"]])]


# ── Email helpers ──────────────────────────────────────────────────────────
def _stage_failure_email(stage_name: str, extra_context: str = "") -> str:
    """Return the HTML body for a per-stage failure email."""
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;">
      <h2 style="color:#c0392b;border-bottom:2px solid #c0392b;padding-bottom:8px;">
        ❌ Stage Failed — Heart Disease Pipeline
      </h2>
      <div style="background:#fdf0f0;border-left:4px solid #c0392b;
                  padding:12px;border-radius:4px;margin-bottom:16px;">
        <p style="margin:0 0 6px;font-size:16px;font-weight:bold;">
          Stage: <code>{stage_name}</code> failed
        </p>
        <p style="color:#555;margin:0;">
          The <b>{stage_name}</b> step of the Heart Disease retraining pipeline
          has failed. Subsequent stages will be skipped.
        </p>
        {f'<p style="color:#888;margin-top:8px;">{extra_context}</p>' if extra_context else ""}
      </div>
      <table style="font-size:13px;color:#333;border-collapse:collapse;width:100%;">
        <tr style="background:#f8f8f8;">
          <td style="padding:6px 10px;font-weight:bold;">DAG</td>
          <td style="padding:6px 10px;">heart_disease_pipeline</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;font-weight:bold;">Stage</td>
          <td style="padding:6px 10px;">{stage_name}</td>
        </tr>
        <tr style="background:#f8f8f8;">
          <td style="padding:6px 10px;font-weight:bold;">Time (UTC)</td>
          <td style="padding:6px 10px;">{{{{ execution_date }}}}</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;font-weight:bold;">Airflow Logs</td>
          <td style="padding:6px 10px;">
            <a href="{{{{ conf.get('webserver', 'base_url') }}}}/dags/heart_disease_pipeline/grid">
              View in Airflow UI
            </a>
          </td>
        </tr>
      </table>
      <p style="font-size:11px;color:#999;margin-top:16px;">
        Heart Disease MLOps Pipeline · Apache Airflow
      </p>
    </div>
    """


# ── Python callables ───────────────────────────────────────────────────────
def _stage_data(**ctx):
    """Move the sensed training file into the shared pipeline volume."""
    src  = Path(TRAINING_DIR) / "train.csv"
    dest = Path(PIPELINE_DIR) / "data" / "raw" / "train.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(src), str(dest))
    ctx["ti"].xcom_push(key="train_path", value=str(dest))
    print(f"Staged: {src} → {dest}")


def _check_model_registered(**ctx):
    """Return task id to branch to after training."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod = [v for v in versions if v.current_stage == "Production"]
        if prod:
            print(f"Production model found: v{prod[0].version}")
            return "restart_services"
        raise ValueError("No production model found")
    except Exception as e:
        raise RuntimeError(f"Model not in registry: {e}")


def _restart_services(**ctx):
    """Gracefully restart the existing MLflow Serve and Streamlit containers."""
    import docker
    client = docker.from_env()
    for container_name in ["hd_mlflow_serve", "hd_streamlit"]:
        try:
            container = client.containers.get(container_name)
            container.restart()
            print(f"Restarted {container_name}")
        except Exception as e:
            print(f"Failed to restart {container_name}: {e}")


def _collect_metrics(**ctx) -> dict:
    """Read all three metrics.json files and push to XCom for the success email."""
    import json
    results = {}
    for model in ["realmlp", "xgboost", "sklearn"]:
        path = Path(PIPELINE_DIR) / "outputs" / model / "metrics.json"
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)
        else:
            results[model] = {}
    ctx["ti"].xcom_push(key="metrics", value=results)
    return results


def _build_success_email_body(**ctx) -> str:
    """Build the HTML success email body with metrics pulled from XCom."""
    metrics = ctx["ti"].xcom_pull(task_ids="collect_metrics", key="metrics") or {}

    rows = ""
    best_model, best_auc = "—", 0.0
    for model, m in metrics.items():
        auc  = m.get("overall_oof_auc", "—")
        ap   = m.get("overall_oof_ap",  "—")
        mean = m.get("mean_fold_auc",   "—")
        std  = m.get("std_fold_auc",    "—")
        run  = m.get("mlflow_run_id",   "—")
        if isinstance(auc, float) and auc > best_auc:
            best_auc, best_model = auc, model
        rows += f"""
        <tr style="background:{'#f0fff4' if model == best_model else 'white'};">
          <td style="padding:7px 10px;font-weight:bold;">
            {model.upper()} {'🏆' if model == best_model else ''}
          </td>
          <td style="padding:7px 10px;">{auc}</td>
          <td style="padding:7px 10px;">{ap}</td>
          <td style="padding:7px 10px;">{mean} ± {std}</td>
          <td style="padding:7px 10px;font-family:monospace;font-size:11px;">{str(run)[:8]}…</td>
        </tr>
        """

    body = f"""
    <div style="font-family:Arial,sans-serif;max-width:640px;">
      <h2 style="color:#27ae60;border-bottom:2px solid #27ae60;padding-bottom:8px;">
        ✅ Pipeline Complete — Heart Disease MLOps
      </h2>
      <p style="color:#555;">
        The full retraining pipeline finished successfully.
        The best model (<b>{best_model.upper()}</b>, OOF AUC = <b>{best_auc:.6f}</b>)
        has been promoted to Production in MLflow and the serving containers
        have been restarted.
      </p>

      <h3 style="margin-top:20px;">Model Comparison</h3>
      <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <thead>
          <tr style="background:#2ecc71;color:white;">
            <th style="padding:7px 10px;text-align:left;">Model</th>
            <th style="padding:7px 10px;">OOF AUC</th>
            <th style="padding:7px 10px;">OOF AP</th>
            <th style="padding:7px 10px;">Mean Fold AUC ± Std</th>
            <th style="padding:7px 10px;">Run ID</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>

      <h3 style="margin-top:20px;">Pipeline Stages</h3>
      <table style="border-collapse:collapse;width:100%;font-size:13px;">
        <tr style="background:#f8f8f8;">
          <td style="padding:6px 10px;">📂 Data staged</td>
          <td style="padding:6px 10px;color:green;">✅ Done</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;">⚙️ DVC preprocess</td>
          <td style="padding:6px 10px;color:green;">✅ Done</td>
        </tr>
        <tr style="background:#f8f8f8;">
          <td style="padding:6px 10px;">🏋️ Training (RealMLP / XGBoost / Sklearn)</td>
          <td style="padding:6px 10px;color:green;">✅ Done</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;">🏆 Best model → MLflow Production</td>
          <td style="padding:6px 10px;color:green;">✅ Done</td>
        </tr>
        <tr style="background:#f8f8f8;">
          <td style="padding:6px 10px;">🔄 Serving containers restarted</td>
          <td style="padding:6px 10px;color:green;">✅ Done</td>
        </tr>
      </table>

      <p style="margin-top:16px;font-size:13px;">
        🔗 <a href="http://localhost:5000">MLflow UI</a> &nbsp;|&nbsp;
        🔗 <a href="http://localhost:8501">Streamlit Dashboard</a> &nbsp;|&nbsp;
        🔗 <a href="http://localhost:3000">Grafana</a>
      </p>
      <p style="font-size:11px;color:#999;margin-top:12px;">
        Heart Disease MLOps Pipeline · Apache Airflow ·
        Run: {{{{ execution_date }}}}
      </p>
    </div>
    """
    ctx["ti"].xcom_push(key="success_email_body", value=body)
    return body


# ── DAG definition ─────────────────────────────────────────────────────────
with DAG(
    dag_id          = "heart_disease_pipeline",
    default_args    = DEFAULT_ARGS,
    description     = "File sensor → Docker DVC pipeline → Restart Services",
    schedule        = timedelta(hours=1),
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    max_active_runs = 1,
    tags            = ["mlops", "heart-disease", "docker"],
) as dag:

    # ── 1. Sense new training file ─────────────────────────────────────────
    sense_file = FileSensor(
        task_id       = "sense_training_file",
        filepath      = str(Path(TRAINING_DIR) / "train.csv"),
        fs_conn_id    = "fs_default",
        poke_interval = 60,
        timeout       = 3600,
        mode          = "poke",
        soft_fail     = False,
    )

    # ── 2. Stage data ──────────────────────────────────────────────────────
    stage_data = PythonOperator(
        task_id         = "stage_data",
        python_callable = _stage_data,
    )

    email_stage_data_failed = EmailOperator(
        task_id      = "email_stage_data_failed",
        to           = ALERT_EMAIL,
        subject      = "❌ [Heart Disease Pipeline] Stage Failed: stage_data",
        html_content = _stage_failure_email(
            "stage_data",
            "Failed to copy train.csv from the watch directory into the pipeline volume."
        ),
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 3a. DVC preprocess ─────────────────────────────────────────────────
    dvc_ingest = DockerOperator(
        task_id      = "run_dvc_ingest",
        image        = "hd_pipeline_image:latest",
        command      = "bash -c 'dvc repro ingest'",
        docker_url   = "unix://var/run/docker.sock",
        network_mode = DOCKER_NETWORK,
        mounts       = DVC_MOUNTS,
        working_dir  = "/opt/pipeline",
        environment  = ENV_VARS,
        mount_tmp_dir = False,
        auto_remove  = "force",
        device_requests = GPU_REQUEST,
    )
    dvc_featurize = DockerOperator(
        task_id      = "run_dvc_featurize",
        image        = "hd_pipeline_image:latest",
        command      = "bash -c 'dvc repro featurize'",
        docker_url   = "unix://var/run/docker.sock",
        network_mode = DOCKER_NETWORK,
        mounts       = DVC_MOUNTS,
        working_dir  = "/opt/pipeline",
        environment  = ENV_VARS,
        mount_tmp_dir = False,
        auto_remove  = "force",
        device_requests = GPU_REQUEST,
    )
    dvc_preprocess = DockerOperator(
        task_id      = "run_dvc_preprocess",
        image        = "hd_pipeline_image:latest",
        command      = "bash -c 'dvc repro preprocess'",
        docker_url   = "unix://var/run/docker.sock",
        network_mode = DOCKER_NETWORK,
        mounts       = DVC_MOUNTS,
        working_dir  = "/opt/pipeline",
        environment  = ENV_VARS,
        mount_tmp_dir = False,
        auto_remove  = "force",
        device_requests = GPU_REQUEST,
    )

    email_preprocess_failed = EmailOperator(
        task_id      = "email_preprocess_failed",
        to           = ALERT_EMAIL,
        subject      = "❌ [Heart Disease Pipeline] Stage Failed: dvc preprocess",
        html_content = _stage_failure_email(
            "run_dvc_preprocess",
            "DVC preprocess stage failed. Check feature engineering in preprocess.py."
        ),
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 3b. DVC train (all three models in parallel) ───────────────────────
    dvc_train = DockerOperator(
        task_id       = "run_dvc_train",
        image         = "hd_pipeline_image:latest",
        command       = "bash -c 'dvc repro train_xgboost train_sklearn train_realmlp'",
        docker_url    = "unix://var/run/docker.sock",
        network_mode  = DOCKER_NETWORK,
        mounts        = DVC_MOUNTS,
        working_dir   = "/opt/pipeline",
        environment   = ENV_VARS,
        mount_tmp_dir = False,
        auto_remove   = "force",
        device_requests = GPU_REQUEST,
    )

    email_train_failed = EmailOperator(
        task_id      = "email_train_failed",
        to           = ALERT_EMAIL,
        subject      = "❌ [Heart Disease Pipeline] Stage Failed: dvc train",
        html_content = _stage_failure_email(
            "run_dvc_train",
            "One or more of train_realmlp / train_xgboost / train_sklearn failed. "
            "Check MLflow for partial runs and DVC logs for the error."
        ),
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 3c. DVC serve_best ─────────────────────────────────────────────────
    # dvc_serve_best = DockerOperator(
    #     task_id       = "run_dvc_serve_best",
    #     image         = "hd_pipeline_image:latest",
    #     command       = "bash -c 'dvc repro serve_best'",
    #     docker_url    = "unix://var/run/docker.sock",
    #     network_mode  = DOCKER_NETWORK,
    #     mounts        = DVC_MOUNTS,
    #     working_dir   = "/opt/pipeline",
    #     environment   = ENV_VARS,
    #     mount_tmp_dir = False,
    #     auto_remove   = "force",
    #     device_requests = GPU_REQUEST,
    # )

    # email_serve_best_failed = EmailOperator(
    #     task_id      = "email_serve_best_failed",
    #     to           = ALERT_EMAIL,
    #     subject      = "❌ [Heart Disease Pipeline] Stage Failed: serve_best",
    #     html_content = _stage_failure_email(
    #         "run_dvc_serve_best",
    #         "serve_best.py failed to compare models or promote the winner to "
    #         "MLflow Production. Check the MLflow registry at http://localhost:5000."
    #     ),
    #     trigger_rule = TriggerRule.ONE_FAILED,
    # )

    # ── 4. Verify model registered ─────────────────────────────────────────
    check_model = BranchPythonOperator(
        task_id         = "check_model_registered",
        python_callable = _check_model_registered,
    )

    email_check_model_failed = EmailOperator(
        task_id      = "email_check_model_failed",
        to           = ALERT_EMAIL,
        subject      = "❌ [Heart Disease Pipeline] Stage Failed: check_model_registered",
        html_content = _stage_failure_email(
            "check_model_registered",
            "No Production model was found in the MLflow registry after training. "
            "serve_best.py may not have run correctly."
        ),
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 5. Restart services ────────────────────────────────────────────────
    restart_services = PythonOperator(
        task_id         = "restart_services",
        python_callable = _restart_services,
    )

    email_restart_failed = EmailOperator(
        task_id      = "email_restart_failed",
        to           = ALERT_EMAIL,
        subject      = "⚠️ [Heart Disease Pipeline] Stage Failed: restart_services",
        html_content = _stage_failure_email(
            "restart_services",
            "hd_mlflow_serve or hd_streamlit failed to restart. "
            "The new model is registered but may not be serving yet. "
            "Run: docker compose restart mlflow-serve streamlit"
        ),
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 6. Collect metrics for success email ───────────────────────────────
    collect_metrics = PythonOperator(
        task_id         = "collect_metrics",
        python_callable = _collect_metrics,
        trigger_rule    = TriggerRule.ALL_SUCCESS,
    )

    # ── 7. Build success email body ────────────────────────────────────────
    build_success_body = PythonOperator(
        task_id         = "build_success_email_body",
        python_callable = _build_success_email_body,
    )

    # ── 8. Send success email ──────────────────────────────────────────────
    email_pipeline_success = EmailOperator(
        task_id      = "email_pipeline_success",
        to           = ALERT_EMAIL,
        subject      = "✅ [Heart Disease Pipeline] Retraining Complete",
        html_content = "{{ ti.xcom_pull(task_ids='build_success_email_body', key='success_email_body') }}",
    )

    # ── 9. Send pipeline-level failure email (any critical stage failed) ───
    email_pipeline_failed = EmailOperator(
        task_id      = "email_pipeline_failed",
        to           = ALERT_EMAIL,
        subject      = "🚨 [Heart Disease Pipeline] Pipeline FAILED",
        html_content = """
        <div style="font-family:Arial,sans-serif;max-width:600px;">
          <h2 style="color:#c0392b;border-bottom:2px solid #c0392b;padding-bottom:8px;">
            🚨 Pipeline Failed — Heart Disease MLOps
          </h2>
          <div style="background:#fdf0f0;border-left:4px solid #c0392b;
                      padding:12px;border-radius:4px;">
            <p style="font-size:15px;font-weight:bold;margin:0 0 8px;">
              The heart disease retraining pipeline did not complete successfully.
            </p>
            <p style="color:#555;margin:0;">
              One or more stages failed. The model in Production has <b>not</b>
              been updated. Check the per-stage failure emails for details, or
              view the full run in Airflow.
            </p>
          </div>
          <p style="margin-top:16px;font-size:13px;">
            🔗 <a href="{{ conf.get('webserver', 'base_url') }}/dags/heart_disease_pipeline/grid">
              View failed run in Airflow UI
            </a>
          </p>
          <table style="font-size:13px;color:#333;margin-top:12px;
                        border-collapse:collapse;width:100%;">
            <tr style="background:#f8f8f8;">
              <td style="padding:6px 10px;font-weight:bold;">DAG</td>
              <td style="padding:6px 10px;">heart_disease_pipeline</td>
            </tr>
            <tr>
              <td style="padding:6px 10px;font-weight:bold;">Run time (UTC)</td>
              <td style="padding:6px 10px;">{{ execution_date }}</td>
            </tr>
            <tr style="background:#f8f8f8;">
              <td style="padding:6px 10px;font-weight:bold;">MLflow</td>
              <td style="padding:6px 10px;">
                <a href="http://localhost:5000">localhost:5000</a>
              </td>
            </tr>
          </table>
          <p style="font-size:11px;color:#999;margin-top:16px;">
            Heart Disease MLOps Pipeline · Apache Airflow
          </p>
        </div>
        """,
        trigger_rule = TriggerRule.ONE_FAILED,
    )

    # ── 10. Log completion (always runs) ───────────────────────────────────
    pipeline_done = BashOperator(
        task_id      = "pipeline_complete",
        bash_command = (
            "echo '=== Pipeline complete ===' && "
            f"cat {PIPELINE_DIR}/outputs/sklearn/metrics.json  || echo 'No sklearn metrics' && "
            f"cat {PIPELINE_DIR}/outputs/realmlp/metrics.json  || echo 'No realmlp metrics' && "
            f"cat {PIPELINE_DIR}/outputs/xgboost/metrics.json  || echo 'No xgboost metrics' && "
            "echo ''"
        ),
        trigger_rule = TriggerRule.ALL_DONE,
    )

    # ── Dependencies ───────────────────────────────────────────────────────

    # Stage data
    sense_file >> stage_data
    stage_data >> email_stage_data_failed

    # Preprocess
    stage_data >> dvc_ingest
    dvc_ingest >> dvc_featurize
    dvc_featurize >> dvc_preprocess
    dvc_preprocess >> email_preprocess_failed

    # Train
    dvc_preprocess >> dvc_train
    dvc_train >> email_train_failed

    # Serve best
    dvc_train >> check_model
    # dvc_serve_best >> email_serve_best_failed

    # Check model
     
    check_model >> email_check_model_failed

    # Restart services
    check_model >> restart_services
    restart_services >> email_restart_failed

    # Success path
    restart_services >> collect_metrics >> build_success_body >> email_pipeline_success

    # Pipeline-level failure email (fires if ANY upstream task failed)
    [
        email_stage_data_failed,
        email_preprocess_failed,
        email_train_failed,
        
        email_check_model_failed,
        email_restart_failed,
    ] >> email_pipeline_failed

    # Always-run completion log
    [email_pipeline_success, email_pipeline_failed] >> pipeline_done