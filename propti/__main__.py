def main():
    import argparse
    import sys
    commands = ["analyse","prepare","run","sampler","sense","sjob"]

    command =  sys.argv[1] if len(sys.argv) > 1 else None
    if command in commands:
        sys.argv.pop(1)
        if command == "analyse":
            from .run import propti_analyse
        elif command == "prepare":
            from .run import propti_prepare
        elif command == "run":
            from .run import propti_run
        elif command == "sampler":
            from .run import propti_sampling
        elif command == "sense":
            from .run import propti_sense
        elif command == "sjob":
            from .run import propti_sjob
    else:
        parser = argparse.ArgumentParser(
                            prog='propti',
                            description='modelling (or optimisation) of parameters in computer simulation with focus on handling the communication between simulation software and optimisation algorithms',
                            epilog="use: 'propti <command> -h' for more information about the sub program.")
        parser.add_argument("command",choices=commands)
        parser.add_argument("args", nargs="*")
        parser.parse_args()