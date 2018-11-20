#! /bin/bash
cd `dirname $0`

if [ ! -f "train-images-idx3-ubyte" ];then
    echo "Downloading train label..."
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    gzip -d train-images-idx3-ubyte.gz
    else
    echo "File train-images-idx3-ubyte is exits"
fi

if [ ! -f "train-labels-idx1-ubyte" ];then
    echo "Downloading train label..."
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gzip -d train-labels-idx1-ubyte
    else
    echo "train label file train-labels-idx1-ubyte is exits"
fi

if [ ! -f "t10k-images-idx3-ubyte" ];then
    echo "Downloading test data..."
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    gzip -d t10k-images-idx3-ubyte
    else
    echo "test data file t10k-images-idx3-ubyte is exits"
fi


if [ ! -f "t10k-labels-idx1-ubyte" ];then
    echo "Downloading test label..."
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gzip -d t10k-labels-idx1-ubyte.gz
    else
    echo "test label file t10k-labels-idx1-ubyte is exits"
fi
