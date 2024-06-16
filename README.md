# Bayesian Differentiable Physics for Cloth Digitalization
Our CVPR 2024 paper [Bayesian Differentiable Physics for Cloth Digitalization](https://arxiv.org/abs/2402.17664). 

![Digitialize Real Fabrics from Cusick Drape Testing Results](images/teaser_with_real.png)

[<img src="images/video_icon.svg" width="50" height="50">](https://youtu.be/ProN0y1bURY?si=p7kFYZA04UZ_zV1y)

## Needed Compilers and Libraries

- GCC 9.5.0 (or MSVC 19.29.30139)
- CUDA 11.3
- Python 3.8.13
- PyTorch 1.12.1
- Kaolin 0.12.0
- Alglib 3.17.0
- Boost 1.75
- Eigen 3.3.9

## How to install

1. Install GCC and CUDA, and confirm their environment variables are set correctly.
2. Install Python (Recommond to use Anaconda).
3. Install Pytorch and Kaolin with following their official documentations.
4. Download Alglib, Boost, and Eigen to a diretory you like.
5. Change the Setup.py to make sure the paths are set correctly, i.e. INCLUDE_DIR.append(...).
6. Run `python setup.py install`.
7. Finally, you can confirm our simulator has been successfully installed by executing the following commonds in prompt:

```python
-> python
-> import pytorch
-> import diffsim
```

Check out the python scripts in the folder **experiments** for training our BDP. They have detailed comments for explaining themselves.

## Cusick Drape Dataset

The dataset in given in the folder data. 

## Authors
Authors
Deshan Gong, Ningtao Mao, and He Wang

Deshan Gong, scdg@leeds.ac.uk

He Wang, he_wang@ucl.ac.uk, [Personal website](https://drhewang.com)

Project Webpage: https://drhewang.com/pages/BDP.html

## Citation (Bibtex)
Please cite our paper if you find it useful:

    @InProceedings{Gong_Bayesian_2024,
    author={Deshan Gong, Ningtao Mao and He Wang},
    booktitle={The Conference on Computer Vision and Pattern Recognition (CVPR)},
    title={Bayesian Differentiable Physics for Cloth Digitalization},
    year={2024}}
