import os
import setuptools

base_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(base_dir, "README.md"), 'r', encoding="utf-8") as f:
    long_description = f.read()
with open(os.path.join(base_dir, "propti", "__init__.py"), 'r', encoding="utf-8") as f:
    version = f.readline().split("=")[-1].strip()[1:-1]

data_files = []
def get_data_files(path, data_files):
    for p in os.listdir(path):
        p = os.path.join(path, p)
        if os.path.isdir(p):
            get_data_files(p, data_files)
        else:
            data_files.append(p)
get_data_files("propti/jobs", data_files)
print(data_files)


setuptools.setup(
    name="propti",
    version=version,
    #TODO #author=
    #TODO #author_email=
    description="PROPTI is an interface tool that couples simulation models with algorithms to solve the inverse problem of material parameter estimation in a modular way. It is designed with flexibility in mind and can communicate with arbitrary algorithm libraries and simulation models. Furthermore, it provides basic means of pre- and post-processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FireDynamics/propti",
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    include_package_data=True,
    data_files=data_files,
    packages = ["propti", "propti/lib", "propti/run", "propti/jobs"],
    install_requires=
        [
            "numpy",
            "matplotlib",
            "scipy",
            "pandas",
            "spotpy",
            #"mpi4py", This should be optional since it is not possible to install on every system with out root
        ],
    entry_points={
        'console_scripts': [
            'propti = propti.__main__:main',
        ]
    }
)