
# Requirement package
## python 3.6
## torch > 0.4
## torchvision > 0.2
## matplotlib > 2.1
## pillow > 5.0
## numpy > 1.14
## scikit-image > 0.13


# STEPs for training Mnist from scratch with pytorch

# 1. pre-processing data:

cd YOURPATH/pytorch_mnist_scratch

# 2. downloading MNIST datasets by wget
bash datasets/get_mnist_datasets.sh

# 3. convert MNIST dataset(byte file) to images ; split traning sets and test sets equally (50% respectively) ; balance the different  class for traning and test sets
python datasets/convert2img.py
python datasets/split_train_test.py

# 4. training original model and show the image of the output of convlution and maxpool/  plot acc-epochs ;loss-epochs curves
python train_mnist.py --show-mid-output True --use-ori-net True --lr 0.001

# 5. training modified model (with one convolution and one fc layer) /  plot acc-epochs ;loss-epochs curves
python train_mnist.py

# 6. all visualization plot are saved in visualization folder
