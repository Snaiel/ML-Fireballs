from pathlib import Path

import structlog
from structlog.typing import WrappedLogger


VERSION = "1.0.0"


def module_processor(logger: WrappedLogger, log_method: str, event_dict: dict):
    path = event_dict.pop("pathname")
    module_path = str(Path(path).relative_to(Path(__file__).parents[1]))
    module = module_path.replace("/", ".").replace(".py", "")
    event_dict["module"] = module
    return event_dict


def reorder_event_dict(logger: WrappedLogger, log_method: str, event_dict: dict) -> dict:
    reordered_dict = {}
    reordered_dict["version"] = event_dict.pop("version")
    reordered_dict["timestamp"] = event_dict.pop("timestamp")
    reordered_dict["process_name"] = event_dict.pop("process_name")
    reordered_dict["module"] = event_dict.pop("module")
    reordered_dict["function"] = event_dict.pop("func_name")
    reordered_dict["level"] = event_dict.pop("level")
    reordered_dict["event"] = event_dict.pop("event")

    if "image" in event_dict:
        reordered_dict["image"] = event_dict.pop("image")
    
    for key in sorted(event_dict.keys()):
        reordered_dict[key] = event_dict[key]
    
    return reordered_dict


def get_logger() -> structlog.stdlib.BoundLogger:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.CallsiteParameterAdder(parameters=[
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.PROCESS_NAME
            ]),
            module_processor,
            reorder_event_dict,
            structlog.processors.JSONRenderer(indent=4),
        ]
    )

    structlog.contextvars.bind_contextvars(version=VERSION)

    return structlog.get_logger()