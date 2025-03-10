# PestML

## Reference projects

[How to create a confusion matrix in PyTorch](https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7)

[PyTorch examples github](https://github.com/pytorch/examples)

[PyTorch ImageNet example github](https://github.com/pytorch/examples/tree/main/imagenet)

## How to train

```sh
python3 pytorchstudy/imagenet_confmatrix.py --epochs 100 --pretrained data_dir
```

##  How to evaluate

```sh
python3 pytorchstudy/imagenet_confmatrix.py -e --resume model_best.pth.tar data_dir
```
