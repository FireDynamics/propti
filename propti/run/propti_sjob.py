import argparse
import subprocess
import os
import re

parser = argparse.ArgumentParser(add_help="Create, start and monitor sampler jobs for Slurm.")
subparser = parser.add_subparsers()
create_parser = subparser.add_parser("create", help="create the job files inside the sampler folders")
create_parser.add_argument("--time", default="0-02:00:00", help="the maximal time the job can run in the following format: days-hours:minutes:seconds");
create_parser.set_defaults(mode="create")
start_parser = subparser.add_parser("start", help="start every sampler job (note: the create command had to be run)")
start_parser.set_defaults(mode="start")
start_parser = subparser.add_parser("cancel", help="cancel every running job for this sampler")
start_parser.set_defaults(mode="cancel")
start_parser = subparser.add_parser("info", help="start jobs")
start_parser.set_defaults(mode="info")
parser.add_argument("root_dir", type=str,
                    help="the folder wich contains all sampler folders (note: the folder name usually named like 'sample_[...]' )", default='.')
cmdl_args = parser.parse_args()

root_dir = cmdl_args.root_dir

def get_table_data():
    """
    Opens the CSV table created by the sampler, reads its contents, and extracts relevant data
    such as file_name, name, and a list of ids.

    Returns:
        tuple: A tuple containing three elements - file_name (str), name (str), and ids (list[str]).
            - file_name: The name of the original file (with extension).
            - name: The CHID name for the simulations.
            - ids: The ids of every simulation
    """
    sample_table_path  = os.path.join(cmdl_args.root_dir, "sample_table.csv")
    sample_table_file = open(sample_table_path)

    file_name = sample_table_file.readline()[13:-1]
    name = file_name.replace(".fds","")
    search = re.search("chid\s*=\s*(.*)",sample_table_file.readline())
    if search:
        name = search.group(1)
    ids = []
    for line in sample_table_file:
        if line.startswith("#") or line == "":
            continue
        else:
            ids.append(line.split(",")[0].strip())
    sample_table_file.close()
    return file_name,name, ids


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



if cmdl_args.mode == "create":
    # The The execution script for the simulation. Variables padded with # get replaced by the corresponding value, for the specific simulation.
    job_file_content = """#!/bin/bash
#!/bin/sh
# Name of the Job
#SBATCH --job-name=#NAME#

# On witch device the simulation is run 
#SBATCH --partition=normal

# Maximum time the job can run: days-hours:minutes:seconds
#SBATCH --time=#TIME#

# Number of cores
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=#NODES#

# Output file name
#SBATCH --output=stdout.%j
#SBATCH --error=stderr.%j


cd #PATH#

mkdir ./results
cd ./results

# define FDS input file
FDSSTEM=../

# grep the CHID (used for stop file)
CHID=`sed -n "s/^.*CHID='\\([-0-9a-zA-Z_]*\\)'.*$/\\1/p" < $FDSSTEM*.fds`

# append the start time to file 'time_start'
echo "$SLURM_JOB_ID -- `date`" >> time_start

# handle the signal sent before the end of the wall clock time
function handler_sigusr1
{
  # protocol stopping time
  echo "$SLURM_JOB_ID -- `date`" >> time_stop
  echo "`date` Shell received stop signal"

  # create FDS stop file
  touch $CHID.stop
  
  # as manual stop was triggered, the end of simulation time was
  # not reached, remove flag file 'simulation_time_end'
  rm simulation_time_end
  wait
}

# register the function 'hander_sigusr1' to handle the signal send out
# just before the end of the wall clock time
trap "handler_sigusr1" SIGUSR1

# check for the simulation finished flag file 'simulation_time_end'
# if it is found, just quit
if [ -e simulation_time_end ]; then
    ## simulation has already finished, nothing left to do
    echo "FDS simulation already finished"
    exit 0
fi

# simulation not finished yet
# create flag file to check for reaching simulation end time
touch simulation_time_end

# Load FDS
module use -a /beegfs/larnold/modules
module load FDS

# set the number of OMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# run FDS executable
mpiexec fds $FDSSTEM*.fds & wait

# TODO make this more robust
# set RESTART to TRUE in FDS input file
sed -i 's/RESTART\s*=\s*(F|.FALSE.)/RESTART=.TRUE./g' $FDSSTEM*.fds

# remove the stop file, otherwise the following chain parts
# would not run
rm $CHID.stop
    """


    fds_file_name, name, ids = get_table_data()
    for id in ids:
        directory =os.path.join(root_dir,f"sample_{id}")
        fds_file_path = os.path.join(directory, fds_file_name)
        job_file_path = os.path.join(directory, "job.sh")
        processes = 1
        fds_file = open(fds_file_path)
        for line in fds_file:
            line = line.strip()
            if line.startswith("&MESH"):
                search = re.search("MPI_PROCESS\s*=\s*([0-9]+)", line)
                if search:
                    processes = int(search.group(1)) + 1
        process_name = f"{name}_{id}"
        fds_file.close()
        job_file = open(job_file_path, "w")
        job = job_file_content.replace("#NAME#", process_name).replace("#TIME#", cmdl_args.time).replace("#NODES#", str(processes)).replace("#PATH#", f"\"{directory}\"")
        job_file.write(job)
        job_file.close()

elif cmdl_args.mode == "start":
    fds_file_name, name, ids = get_table_data()
    for id in ids:
        job_file_path = os.path.abspath(os.path.join(root_dir,f"sample_{id}", "job.sh"))
        output = subprocess.check_output(["sbatch", job_file_path])
elif cmdl_args.mode == "cancel":
    output = subprocess.check_output(["squeue", "-o", r"\"%o;%i\""])
    root_path = str.encode("\\\"" + os.path.abspath(""))
    print(root_path)
    for line in output.splitlines():
        if line.startswith(root_path):
            line = line[2:-3].decode("utf-8") 
            (_, id) = line.split(";")
            subprocess.check_output(["scancel", id])

elif cmdl_args.mode == "info":
    def text(t, c):
        l = len(t)
        if l > c:
            return t[l - c:]
        else:
            return " " * (c - l) + t

    paths = []
    fds_file_name, name, ids = get_table_data()
    for id in ids:
        job_file_path = os.path.abspath(os.path.join(root_dir,f"sample_{id}", "job.sh"))
        paths.append((job_file_path, id))
    total = len(paths)

    output = subprocess.check_output(["squeue", "-o", r"\"%o;%i;%j;%t;%M;%l\""])

    root_path = str.encode("\\\"" + os.path.abspath(""))
    print(root_path)
    id, job_name, status, time, time_limit = "JOBID","NAME", "ST", "TIME", "TIME_LIMIT"
    print(f"{text(job_name,30)} | {text(id,8)} | {text(status,2)} | {text(time,10)} | {text(time_limit,10)}")
    for line in output.splitlines():
        if line.startswith(root_path):
            line = line[2:-3].decode("utf-8") 
            (job_path, id, job_name, status, time, time_limit) = line.split(";")
            print(f"{text(job_name,30)} | {text(id,8)} | {text(status,2)} | {text(time,10)} | {text(time_limit,10)}")
            for i in range(len(paths)):
                if paths[i][0] == job_path:
                    paths.pop(i)
                    break
    
    error = 0
    for (path, id) in paths:
        info_path =  os.path.join(os.path.dirname(path),"results",f"{name}.out")
        if os.path.exists(info_path):
            reader = reverse_readline(info_path)
            line = next(reader, "")
            if "FDS completed successfully" in line:
                job_name, status = name + "_" + id, "F"
            else: 
                job_name, status = name + "_" + id, "E " # Indent for easier spotting
                error += 1
        else: 
            job_name, status = name + "_" + id, "E " # Indent for easier spotting
            error += 1
        print(f"{text(job_name,30)} | -------- | {text(status,2)} | ---------- | ----------")
    if error == 1:
        print(f"{len(paths)}/{total} finished with {error} error.")
    else:
        print(f"{len(paths)}/{total} finished with {error} errors.")