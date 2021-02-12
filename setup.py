import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fpgaconvnet-optimiser-AlexMontgomerie", # Replace with your own username
    version="0.0.3",
    author="Alex Montgomerie",
    author_email="am9215@ic.ac.uk",
    description="Optimiser for mapping convolutional neural network models to FPGA platforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexMontgomerie/fpgaconvnet-optimiser",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "networkx==2.5",
        "numpy==1.19.2",
        "protobuf==3.14.0",
        "torch==1.7.1",
        "pyyaml==5.3.1",
        "scipy==1.5.2",
        "torchvision==0.8.2",
        "onnx>=1.8.0",
        "onnxruntime==1.6.0"
    ]
)
