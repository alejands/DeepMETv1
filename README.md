## DeepMETv1

DeepMETv1 is a fully-connected neural network (FCNN) for MET reconstruction in CMS data. This branch is focused on training on Run3 conditions.

The [original repository](https://github.com/yongbinfeng/DeepMETTraining) by [Yongbing Feng](https://github.com/yongbinfeng) was used to train DeepMETv1 on Run2 conditions. A copy of this original code is in the branch [Run2](https://github.com/DeepMETv2/DeepMETv1/tree/Run2).

---

Install the necessary packages with [MiniConda](https://docs.conda.io/en/latest/miniconda.html). You can use the provided `deepmetv1_env.yml` file.
conda env create -y -f deepmetv1_env.yml
```

Activate the environment

```
conda activate METTraining
```

Prepare the `h5` training files
```
python convertNanoToHDF5.py -i /eos/cms/store/user/yofeng/WRecoilNanoAOD_Skimmed_v10_tempMET_TTJets/myNanoProdMc2016_NANO_2_Skim.root -o /eos/cms/store/user/yofeng/DeepMETTrainingFile/myNanoProdMc2016_NANO_ttbar_2_Skim.h5
```

Run the training
```
python train_ptmiss_mine.py -i input.txt
```
