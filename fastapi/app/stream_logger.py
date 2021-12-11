from logging import INFO, Formatter, StreamHandler, getLogger

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def of(name):
    logger = getLogger(name)

    handler = StreamHandler()
    handler.setFormatter(Formatter(LOG_FORMAT))
    handler.setLevel(INFO)

    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
