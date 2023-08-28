import argparse
import pickle
import subprocess
import os
import sys

parser = argparse.ArgumentParser(add_help="Create, start and monitor sampler jobs.")
subparser = parser.add_subparsers()

create_parser = subparser.add_parser("create", help="create the job files inside the sampler folders")
create_parser.add_argument("--scheduler", help="select wich scheduler is used", choices=["slurm"], type=str)
create_parser.add_argument("--template", help="define a path to a template", type=str)
create_parser.add_argument("--parameters", help="define parameters wich get replaced in the template separate every parameter by ';' \na parameter should be described with 'name' to get a value from the sample \nor 'name,value' to set a constant value.", type=str)
create_parser.set_defaults(mode="create")

start_parser = subparser.add_parser("start", help="start every sampler job.")
start_parser.set_defaults(mode="start")

cancel_parser = subparser.add_parser("cancel", help="cancel every running job for this sampler.")
cancel_parser.set_defaults(mode="cancel")

info_parser = subparser.add_parser("info", help="display information about teh status of every job for this sampler.")
info_parser.set_defaults(mode="info")

template_parser = subparser.add_parser("template", help="clone a template.")
template_parser.add_argument("--name", help="name of the template", type=str, choices=["fds"])
template_parser.set_defaults(mode="template")

cmdl_args = parser.parse_args()



def load_from_picke_file():
    # Check if `propti.pickle.sampler` exists
    if not os.path.isfile('propti.pickle.sampler'):
        sys.exit("'propti.pickle.sampler' not detected. Script execution stopped.")

    in_file = open("propti.pickle.sampler", 'rb')
    (sampler_data, job) = pickle.load(in_file)
    return sampler_data, job


# Reading a complete file when only the last line is important is inefficient,
# therefore a helper function is created that reads a file in reverse.
# source: https://stackoverflow.com/questions/2301789/how-to-read-a-file-in-reverse-order
def reverse_readline(filename, buf_size=8192):
    """
    A generator that returns the lines of a file in reverse order
    """
    with open(filename, 'rb') as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size)).decode(encoding='utf-8')
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment

def scheduler_not_supported(name):
    sys.exit(f"scheduler {name} is not supported")

if cmdl_args.mode == "create":
    (sampler_data, job) = load_from_picke_file()

    from .. import lib as pr

    parameters = []
    for p in cmdl_args.parameter.split(";"):
        s = p.split(",")
        if len(s) == 1:
            parameters.append(s[0])
        elif len(s) == 2:
            (name, value) = s
            parameters.append((name, value))
        else:
            name = s.pop(0)
            parameters.append((name, s))

    job = pr.Job(cmdl_args.scheduler, cmdl_args.template, parameters)

    for (execution_dirs, sample_sets) in sampler_data:
        job.create_jobs(execution_dirs, sample_sets)

    out_file = open('propti.pickle.sampler', 'wb')
    pickle.dump((sampler_data, job), out_file)
    out_file.close()

elif cmdl_args.mode == "start":
    (sampler_data, job) = load_from_picke_file()
    if job == None:
        sys.exit("Job settings are missing. Use 'create' to create jobs oder define it inside the sampler input file")
    scheduler = job.scheduler
    if scheduler == "slurm":
        for (execution_dirs, _) in sampler_data:
            for execution_dir in execution_dirs:
                execution_dir = os.path.abspath(execution_dir)
                output = subprocess.check_output(["sbatch",  "job.sh"], cwd=execution_dir)
    else:
        scheduler_not_supported(scheduler)
        pass
elif cmdl_args.mode == "cancel":
    (_, job) = load_from_picke_file()
    scheduler = job.scheduler
    if scheduler == "slurm":
        output = subprocess.check_output(["squeue", "-o", r"\"%o;%i\""])
        root_path = str.encode("\\\"" + os.path.abspath(""))
        for line in output.splitlines():
            if line.startswith(root_path):
                line = line[2:-3].decode("utf-8") 
                (_, job_id) = line.split(";")
                subprocess.check_output(["scancel", job_id])
    else:
        scheduler_not_supported(scheduler)
        pass

elif cmdl_args.mode == "info":
    (sampler_data, job) = load_from_picke_file()
    scheduler = job.scheduler
    if scheduler == "slurm":
        def text(t: str, c: int):
            """
            Helper function to align test to the left side.
            """
            l = len(t)
            if l > c:
                return t[l - c:]
            else:
                return " " * (c - l) + t

        paths = []
        for (execution_dirs, _) in sampler_data:
            for execution_dir in execution_dirs:
                job_file_path = os.path.abspath(os.path.join(execution_dir, "job.sh"))
                paths.append(job_file_path)
        total = len(paths)

        output = subprocess.check_output(["squeue", "-o", r"\"%o;%i;%j;%t;%M;%l\""])


        root_path = str.encode("\\\"" + os.path.abspath(""))
        job_id, job_name, status, time, time_limit = "JOBID", "NAME", "ST", "TIME", "TIME_LIMIT"
        print(f"{text(job_name,30)} | {text(job_id,8)} | {text(status,2)} | {text(time,10)} | {text(time_limit,10)}")
        for line in output.splitlines():
            if line.startswith(root_path):
                line = line[2:-2].decode("utf-8") 
                (job_path, job_id, _, status, time, time_limit) = line.split(";")
                for i in range(len(paths)):
                    if paths[i] == job_path:
                        paths.pop(i)
                        break
                job_name = os.path.basename(os.path.dirname(job_path))
                print(f"{text(job_name,30)} | {text(job_id,8)} | {text(status,2)} | {text(time,10)} | {text(time_limit,10)}")
        
        error = 0
        for path in paths:
            job_name = os.path.basename(os.path.dirname(path))
            info_path = None
            for name in os.listdir(os.path.dirname(path)):
                if name.startswith("stderr"):
                    info_path = os.path.join(os.path.dirname(path),name)
            if info_path != None:
                reader = reverse_readline(info_path)
                line = next(reader, "") + next(reader, "")
                if "FDS completed successfully" in line:
                    print(f"{text(job_name,30)} | -------- |  F | ---------- | ----------")
                    continue
            error += 1
            print(f"{text(job_name,30)} | -------- | E  | ---------- | ----------")
        if error == 1:
            print(f"{len(paths)}/{total} finished with {error} error.")
        else:
            print(f"{len(paths)}/{total} finished with {error} errors.")
    else:
        scheduler_not_supported(scheduler)
        pass

elif cmdl_args.mode == "template":
    path = os.path.join("/".join(__loader__.path.split("/")[:-2]), "jobs", cmdl_args.name)
    if os.path.exists(path):
        with open(path) as read:
            with open("job_template","w") as write:
                write.write(read.read())
        print("Copied template as 'job_template'.")
    else:
        sys.exit(f"A template with the name '{cmdl_args.name}' does not exist")

