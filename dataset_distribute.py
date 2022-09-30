from operator import ne
import os
root_path = '/mnt/e/Dataset/satellite_0415'
old_img_path = os.path.join(root_path, 'images')
new_img_path = os.path.join(root_path, 'JPEGImages')
old_labels_path = os.path.join(root_path, 'labels')
new_labels_path = os.path.join(root_path, 'Annotations')
os.rename(old_img_path, new_img_path)
os.rename(old_labels_path, new_labels_path)
files = os.listdir(os.path.join(root_path, 'Annotations'))
train_val = []
tv = ''
test = []
ts = ''
train = []
tr = ''
val = []
v = ''
for i in files:
    a = i.split('.')
    if os.path.exists(os.path.join(root_path, 'JPEGImages', a[0]+'.tif')):
        if hash(a[0])%10 >=1:
            train_val.append(a[0])
            tv += a[0] + '\n'
            if hash(a[0])%10 >=3:
                train.append(a[0])
                tr += a[0] + '\n'
            else:
                val.append(a[0])
                v += a[0] + '\n'
        else:
            test.append(a[0])
            ts += a[0] + '\n'
    else:
        print(os.path.join(root_path, 'JPEGImages', a[0]+'.tif') + "does not exist")
        # os.remove(os.path.join(root_path, 'labels', a[0]+'.xml'))
        
output_path = os.path.join(root_path, 'ImageSets/Main/')
print(output_path)
if os.path.exists(output_path):
    pass
else:
    os.makedirs(output_path)
    
wf=open(os.path.join(output_path, "train.txt"),'w+',encoding='utf-8')
wf.write(tr)
wf.close()

wf=open(os.path.join(output_path, "trainval.txt"),'w+',encoding='utf-8')
wf.write(tv)
wf.close()

wf=open(os.path.join(output_path, "val.txt"),'w+',encoding='utf-8')
wf.write(v)
wf.close()

wf=open(os.path.join(output_path, "test.txt"),'w+',encoding='utf-8')
wf.write(ts)
wf.close()

print(f'train {len(train)}')
print(f'val {len(val)}')
print(f'test{len(test)}')
print(f'sum {len(train_val)+len(test)}')