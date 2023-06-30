from .lib import *

__version__ = "1.0.0"

def main():
    import argparse
    import sys

    commands = ["analyse","prepare","run","sense"]

    command =  sys.argv[1] if len(sys.argv) > 1 else None
    if command in commands:
        sys.argv.pop(1)
        if command == "analyse":
            from .run import propti_analyse
        elif command == "prepare":
            from .run import propti_prepare
        elif command == "run":
            from .run import propti_run
        elif command == "sense":
            from .run import propti_sense
    else:
        parser = argparse.ArgumentParser(
                            prog='propti',
                            description='Test calling different sub programs',
                            epilog="use: 'propti <command> -h' for more information about the sub program.")
        parser.add_argument("command",choices=commands)
        parser.add_argument("args", nargs="*")
        parser.parse_args()