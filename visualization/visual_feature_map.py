import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

def plot_mid_layer_output(data, name):
    data = (data - data.min()) / (data.max() - data.min())
    if len(data.shape) > 3:
    	data = np.squeeze(data)
    #elif len(data.shape) == 2:
    #    data = np.expand_dims(data,0)
    #print(data.shape)
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # add some space between filters
    padding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   
    #tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data)
    plt.axis('off')
    plt.savefig('visualization/' + name + '.png', format='png')
