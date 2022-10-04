import logging


def logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    loggerFormat = logging.Formatter("%(asctime)s, %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(loggerFormat)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(loggerFormat)
    logger.addHandler(streamHandler)