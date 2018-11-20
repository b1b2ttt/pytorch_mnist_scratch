import os
from math import floor

root = __file__.split('/')[0]
#root = os.path.split(os.path.realpath(__file__))[0]

with open(root+'/labels.txt','r') as f:
    imgs = []
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0],int(words[1])))

# sort all samples by label
imgs = sorted(imgs, key=lambda x:x[1])
path = [x[0] for x in imgs]
label = [x[1] for x in imgs]

def searchRange(nums, target):
    if len(nums) == 1 and nums[0] == target:
        return [0, 0]
    try:
        index = nums.index(target)
        count = index
        for i in range(index + 1, len(nums)):
            if nums[i] == target:
                count += 1
            else:
                break
        return ([index, count])
    except:
        return ([-1, -1])


# split train and test data equally  and set all classes are roughly equally in training and test
testf = open(root+'/testsets.txt','w+')
trainf = open(root+'/trainsets.txt','w+')
for i in range(0,10):
    range_index =  searchRange(label,i)
    middle = floor( sum(range_index)/2.0 )
    for j in range(range_index[0], middle + 1):
        trainf.write(path[j] + ' ' + str(label[j]) + '\n')
    for j in range( middle + 1, range_index[1] + 1):
        testf.write(path[j] + ' ' + str(label[j]) + '\n')

trainf.close()
testf.close()

