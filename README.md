# Bayesian-Differentiable-Physics-for-Cloth-Digitalization
Our CVPR 2024 paper 'Bayesian Differentiable Physics for Cloth Digitalization'

## Needed Compilers and Libraries

- GCC 9.5.0 (or MSVC)
- CUDA 11.3
- PyTorch 1.12.1
- Kaolin 0.12.0
- Alglib 3.17.0
- Boost 1.75
- Eigen 3.3.9

## How to install

1. Install GCC and CUDA, and confirm their environment variables are set correctly.
2. Install Python (Recommond to use Anaconda to create a new environment).
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

# Cusick Drape Dataset

The data are available in the .\data\... 

More data will be uploaded soon.

# To be cont...
