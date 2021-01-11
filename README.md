# fpgaConvNet Optimiser

## Setup

The following programs are required:

```
python=3.6
onnx=1.8.0
```

The first step is to setup the python environment. It is recommended to use [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment. The following installs an environment called `fpgaconvnet`.

```
conda env create -f environment.yml
```

Instructions for installing [onnx](https://github.com/onnx/onnx) can be found on their github page. The instructions are copied here:

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py install
```

## Optimiser Framework

An example of running the optimiser is as follows:

```
python -m run_optimiser -n vgg16 \
    -m examples/models/vgg16bn.onnx \
    -p examples/platforms/zc706.json \
    -o outputs/vgg16 \
    -b 256 \
    --objective throughput \
    --transforms fine weights_reloading coarse partition \
    --optimiser simulate_annealing
```

### Modelling

### Transforms

| Transform | Level | Descrption |
|-----------|-------|------------|
| Fine | Layer | |
| Coarse | Partition | |
| Weights Reloading | Partition | |
| Partitioning | Network | |

### Optimisation Schemes

Using
The optimisation schemes implemented

## Citations

If you use this work, please use the following references:

```BibTex
@article{venieris_fpgaconvnet_2019,
    title = {fpgaConvNet: Mapping Regular and Irregular Convolutional Neural Networks on FPGAs},
    journal = {IEEE Transactions on Neural Networks and Learning Systems},
    author = {Venieris, S. I. and Bouganis, C.},
    year = {2019},
}

@inproceedings{venieris_fpgaconvnet_2017,
    title = {fpgaConvNet: A Toolflow for Mapping Diverse Convolutional Neural Networks on Embedded FPGAs},
    booktitle = {NIPS 2017 Workshop on Machine Learning on the Phone and other Consumer Devices},
    author = {Venieris, Stylianos I. and Bouganis, Christos-Savvas},
    year = {2017},
}

edings{venieris_fpgaconvnet_2016,
    title = {fpgaConvNet: A Framework for Mapping Convolutional Neural Networks on FPGAs},
    booktitle = {2016 IEEE 24th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
    author = {Venieris, S. I. and Bouganis, C.},
    year = {2016},
}
```

---

Feel free to post an issue if you have any questions or problems!

