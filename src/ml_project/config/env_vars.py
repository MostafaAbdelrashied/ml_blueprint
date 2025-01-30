import os
from dataclasses import dataclass
from typing import List


@dataclass
class EnvCheckResult:
    is_valid: bool
    missing_vars: List[str]
    message: str


def check_environment_variables(*required_vars: str) -> EnvCheckResult:
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        return EnvCheckResult(
            is_valid=False,
            missing_vars=missing_vars,
            message=f"Missing required environment variables: {', '.join(missing_vars)}",
        )

    return EnvCheckResult(
        is_valid=True,
        missing_vars=[],
        message="All required environment variables are set",
    )
