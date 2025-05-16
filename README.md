# HSRMamba
HSRMamba: Contextual Spatial-Spectral State Space Model for Single Hyperspectral Super-Resolution

## Requirements
* Python 3.9.19
* Pytorch 1.13.1
* Pytorch-cuda 11.7
* mamba-ssm 1.0.1
* causal_conv1d 1.0.0

## Preparation
To get the training set, validation set and testing set, refer to SSPSR to download the mcodes for cropping the hyperspectral image.

## Training
To train HSRMamba, run the following command.<br>
```
sh demo.sh
```
## References
* [SSPSR](https://github.com/junjun-jiang/SSPSR)
* [MambaIR](https://github.com/csguoh/MambaIR)
