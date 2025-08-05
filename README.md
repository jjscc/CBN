# Learning a Better SPD Network for Signal Classification: A Riemannian Batch Normalization Method.

# README
This is the official code for our TNNLS publication: Learning a Better SPD Network for Signal Classification: A Riemannian Batch Normalization Method.

In case you have any problem, do not hesitate to contact me 2932723775@qq.com.

### Dataset
We further release our preprocessed  HDM05 dataset.

Please download the [HDM05](https://drive.google.com/file/d/1T6ay9KKzhgM1hg05w8Buefok58MMevYh/view?usp=drive_link) 

### Running experiments
To train and test the experiments on the HDM05 dataset, run this command:

```train and test
python Hdm05.py
```

## Requirements
 - Python == 3.10.13
 - Pytorch == 2.2.1
 - Pytorch == 1.26.4
 - [Geoopt](https://github.com/geoopt/geoopt) == 0.5.0
