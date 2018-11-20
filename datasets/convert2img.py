import os
from skimage import io
import torchvision.datasets.mnist as mnist

print('waiting 30 seconds')
root = __file__.split('/')[0]
#root = os.path.split(os.path.realpath(__file__))[0]
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
#print("training set :",train_set[0].size())
#print("test set :",test_set[0].size())

def convert_to_img():
    with open(root+'/labels.txt','w+') as f:
        data_path=root+'/imgs/'

        if(not os.path.exists(data_path)):
            os.makedirs(data_path)

        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            if (not os.path.exists(img_path)):
                io.imsave(img_path,img.numpy())
                f.write(img_path+' '+str(label.numpy())+'\n')
    
        train_num = train_set[0].size()[0] 
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            index = train_num + i 
            img_path = data_path+ str(index) + '.jpg'
            if (not os.path.exists(img_path)):
                io.imsave(img_path, img.numpy())
                f.write(img_path + ' ' + str(label.numpy()) + '\n')

convert_to_img()
