#TODO

import os
import setuptools
#import propti

base_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(base_dir, "README.md"), 'r', encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="propti",
    version="0.2.0",
    # use_incremental=True,
    # setup_requires=["incremental"],
    # #author=
    # #author_email=
    description="PROPTI is an interface tool that couples simulation models with algorithms to solve the inverse problem of material parameter estimation in a modular way. It is designed with flexibility in mind and can communicate with arbitrary algorithm libraries and simulation models. Furthermore, it provides basic means of pre- and post-processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FireDynamics/propti",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    package_dir={"propti": "propti"},
    install_requires=
        [
            "numpy",
            "matplotlib",
            "scipy",
            "pandas",
            "spotpy",
            "mpi4py",
            # "incremental",
        ],
    entry_points={
        'console_scripts': [
            'propti = propti:main',
        ]
    }
)