## Run propti interactive
```bash
# Linux
docker run -it -v $(pwd):/workdir ghcr.io/FireDynamics/propti

# Windows PowerShell
docker run --it -v ${pwd}:/workdir ghcr.io/FireDynamics/propti

# Windows Command Prompt
docker run -it -v %cd%:/workdir ghcr.io/FireDynamics/propti
```

You can now use propti with the following commands:
- `propti_analyse <...arguments>`
- `propti_prepare <input-filename>`
- `propti_run .`
- `propti_sense .`

You can display the help for each command by using the `-h` argument (e.g. `propti_analyse -h`).

## Run propti non-interactive
```bash
# Linux
docker run --rm -v $(pwd):/workdir ghcr.io/FireDynamics/propti {propti_analyse|propti_prepare|propti_run|propti_sense} <...arguments>

# Windows PowerShell
docker run --rm -v ${pwd}:/workdir ghcr.io/FireDynamics/propti {propti_analyse|propti_prepare|propti_run|propti_sense} <...arguments>

# Windows Command Prompt
docker run --rm -v %cd%:/workdir ghcr.io/FireDynamics/propti {propti_analyse|propti_prepare|propti_run|propti_sense} <...arguments>
```

## Additional information
This docker image is based on the FDS image [ghcr.io/openbcl/fds](https://github.com/openbcl/fds-dockerfiles/pkgs/container/fds).
There you will find all information on options, problems, errors and their solutions when operating FDS in Docker containers.

## How to build the docker image by yourself
1. Clone this repository.
1. Navigate with your shell to this folder.
1. Choose between building the image with the latest FDS version or with a specific one. You can find all available versions (tags) [here](https://github.com/openbcl/fds-dockerfiles/pkgs/container/fds/versions).
For compatibility reasons, the FDS version should be at least 6.7.4.

```bash
# build command: propti image with latest FDS version
docker build -t propti -f Dockerfile ..

# build command: propti image with specific FDS version (e.g. 6.9.1)
docker build --build-arg="FDS_VERSION=6.9.1" -t propti -f Dockerfile .. 
```

To run your docker image of propti use the `docker run` commands described above and replace the package name `ghcr.io/FireDynamics/propti` with `propti`.