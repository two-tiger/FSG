# FSG: Fast and Slow Gradient Approximation for Binary Neural Network Optimization
Codes for Accepted Paper : "Fast and Slow Gradient Approximation for Binary Neural Network Optimization" in AAAI 2025.

## How to use it

### Dependencies
```shell
pip install -r requirements.txt
```

### Prepare pre-trained model
Please check [MetaQuant tutorial](https://github.com/csyhhu/MetaQuant) to train your own pre-trained model.

Or you can use the default pretrained model provided by us. Uploaded in ```Results/model-dataset/model-dataset-pretrain.pth``` 

### Quick Start
The following commands run FSG on ResNet20 using CIFAR10 dataset with dorefa as forward
quantization method and Adam as optimization. 

The Fast-net is Multi-FC and the Slow-net is Mamba

The resulting quantized model is quantized using 1 bits: {+1, -1} for 
all layers (conv, fc). 

Initial learning rate is set as 1e-3 and decreases by a factor of 0.1 every
30 epochs: 1e-3->1e-4->1e-5:

```python3
CUDA_VISIBLE_DEVICES='0' python meta-quantize.py -m ResNet44 -d CIFAR10 -q dorefa -bw 1 -o adam -meta MetaFastAndSlow -hidden 100 -lr 1e-3 -n 100
```

The following commands run FSG on ResNet56 using CIFAR100 dataset with dorefa as forward
quantization method and Adam as optimization. 

```python3
CUDA_VISIBLE_DEVICES='0' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta MetaFastAndSlow -hidden 100 -lr 1e-3 -n 100
```

## Support
Leave an issue if there is any bug and email me if any concerns about paper.

## Citation
Cite the paper if anything helps you:

```angular2

```