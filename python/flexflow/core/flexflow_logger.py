import logging

def setup_custom_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.DEBUG)
  logger.propagate = 0
  if not logger.handlers:
    formatter = logging.Formatter(fmt='%(levelname)s - %(module)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
  return logger
  
fflogger = setup_custom_logger('fflogger')