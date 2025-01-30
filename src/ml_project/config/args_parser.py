import argparse


def parse_args() -> argparse.Namespace:
    """
    Parse the arguments passed to the script
    :return: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="Train and evaluate a model")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Set the logging level",
    )

    args, _ = parser.parse_known_args()

    return args
