from loguru import logger

from ml_project.config.args_parser import parse_args
from ml_project.config.env_vars import check_environment_variables
from ml_project.config.logging import setup_logging

args = parse_args()
setup_logging(args.log_level)

REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "DATA_PATH"]

env_results = check_environment_variables(
    *REQUIRED_ENV_VARS,
)
if not env_results.is_valid:
    logger.error(env_results.message)
    exit(1)
