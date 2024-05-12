# Ambiguity Consistency and Uncertainty Minimization for Semi-Supervised Medical Image Segmentation

### Introduction
This repository is for our paper: 'Ambiguity Consistency and Uncertainty Minimization for Semi-Supervised Medical Image Segmentation'.

## Requirements

Some important required packages include:

* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

### Usage
1. Put the data in './ACUM/data';
3. Train the model;
```
cd code
python train_2d.py --root_path your DATA_PATH
```
4. Test the model;
```
python test_2d.py 
```


### Acknowledgements:
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MCF](https://github.com/WYC-321/MCF). Thanks for these authors for their valuable works.

