# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_evaluate, visualize_results

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
dag = DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='House Price Prediction with Random Forest',
    schedule_interval=None,
    catchup=False,
)

# Task to load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task to perform data preprocessing
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task to build and save model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)

# Task to evaluate the model
evaluate_model_task = PythonOperator(
    task_id='evaluate_model_task',
    python_callable=load_model_evaluate,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)

# Task to create visualizations
visualize_task = PythonOperator(
    task_id='visualize_task',
    python_callable=visualize_results,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)

# Set task dependencies
load_data_task >> data_preprocessing_task >> build_save_model_task >> [evaluate_model_task, visualize_task]

if __name__ == "__main__":
    dag.cli()
