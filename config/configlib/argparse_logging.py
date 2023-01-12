import argparse
import logging

class LogLevel(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        logging.basicConfig(level=getattr(logging, values))

def set_log_level(*args, **kwargs):
    return LogLevel(*args, **kwargs)
