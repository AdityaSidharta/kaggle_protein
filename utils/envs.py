import os

project_path = os.getenv("PROJECT_PATH")
data_source_path = os.path.join(project_path, "data_source")
output_path = os.path.join(project_path, "output")

train_path = os.path.join(data_source_path, "train")
test_path = os.path.join(data_source_path, "test")
sample_submission_path = os.path.join(data_source_path, "sample_submission.csv")
train_repo_path = os.path.join(data_source_path, "train.csv")

model_cp_path = os.path.join(output_path, "model_checkpoint")
logger_path = os.path.join(output_path, "logs")
result_path = os.path.join(output_path, "result")

logger_repo = os.path.join(logger_path, "logger.log")
