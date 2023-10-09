import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

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
    python_requires='>=3.10',
    install_requires=required_packages,
)
