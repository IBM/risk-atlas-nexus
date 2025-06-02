import logging

import logzero


def configure_logger(
    logger_name: str,
    logging_level="INFO",
    json=False,
):
    if logger_name is None or logger_name == "":
        raise Exception(
            "Logger name cannot be None or empty. Accessing root logger is restricted. Please use routine configure_root_logger() to access the root logger."
        )

    logging_level = logging.getLevelNamesMapping()[logging_level.upper()]

    log_format = (
        "%(color)s[%(asctime)s:%(msecs)d] - %(levelname)s - RiskAtlasNexus - "
        "%(end_color)s%(message)s"
    )

    formatter = logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    return logzero.setup_logger(
        name=logger_name, level=logging_level, formatter=formatter, json=json
    )
