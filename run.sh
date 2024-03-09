python full_precision.py -m ResNet56 -d CIFAR100 -o Adam --epoch 500 > out/full_precision.log     

CUDA_VISIBLE_DEVICES='1' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta LSTMFC-Grad -hidden 100 -lr 1e-3 -n 500 > out/lstm-grad.log

CUDA_VISIBLE_DEVICES='1' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta MultiFC -hidden 100 -lr 1e-3 -n 500 > out/multifc.log

CUDA_VISIBLE_DEVICES='1' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta LSTMFC -hidden 100 -lr 1e-3 -n 500 > out/lstm-weight.log

CUDA_VISIBLE_DEVICES='1' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta LSTMFC-merge -hidden 100 -lr 1e-3 -n 500 > out/lstm-merge.log

CUDA_VISIBLE_DEVICES='0' python meta-quantize.py -m ResNet56 -d CIFAR100 -q dorefa -bw 1 -o adam -meta LSTMFC-momentum -hidden 100 -lr 1e-3 -n 500 > out/lstm-momentum.log