from __future__ import absolute_import

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fpgaconvnet-optimiser", # Replace with your own username
    version="0.1.0",
    author="Alex Montgomerie",
    author_email="am9215@ic.ac.uk",
    description="Optimiser for mapping convolutional neural network models to FPGA platforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexMontgomerie/fpgaconvnet-optimiser",
    include_package_data=True,
    packages=setuptools.find_namespace_packages(
        include=['fpgaconvnet.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "toml>=0.10.2",
        "wandb>=0.15.0"
    ]
)
