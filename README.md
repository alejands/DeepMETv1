## DeepMETv1

DeepMETv1 is a fully-connected neural network (FCNN) for MET reconstruction in CMS data. This branch is focused on training on Run3 conditions.

The [original repository](https://github.com/yongbinfeng/DeepMETTraining) by Yongbing Feng was used to train DeepMETv1 on Run2 conditions. A copy of this original code is in the branch [Run2](https://github.com/DeepMETv2/DeepMETv1/tree/Run2).

---

Install the necessary packages with [MiniConda](https://docs.conda.io/en/latest/miniconda.html). You can use the provided `environment.yml` file.

```
conda env create -y -f environment.yml
```

Note: Using `environment.yml` is recommended, but a setup file with explicit package versions `explicit.yml` is also provided.

Activate the environment

```
conda activate METTraining
```

---

Prepare the HDF5 training files (to see log info, add option `-v`)

```
python convertNanoToHDF5.py inputfile.root outputfile.h5
```

To see more options

```
python convertNanoToHDF5.py --help
```

---

Run the training

```
python train_ptmiss_mine.py -i input.txt
```
