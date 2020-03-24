import os
import numpy as np
dur_dir='//data5/wangshiming/my_tacotron2/mask_Synthesizer/0/dur_predict'
dur_predict=list(filter(lambda x:x.endswith('dur'),os.listdir(dur_dir)))
label_dir='/data5/wangshiming/biaobei/biaobei/label'
x=[]
y=[]
for file in dur_predict:

    with open(os.path.join(dur_dir,file),'r',encoding='gbk')as reader:
        lines=reader.readlines()
        for line in lines[1:-1]:
            x.append(int(line.strip().split(' ')[1])*10)
    label=file.replace('.dur','.label')
  
    with open(os.path.join(label_dir,label),'r') as reader:
        lines = reader.readlines()
        lines = lines[1:-1]
        for line in lines:
            start, end, _ = line.strip().split(' ')
            y.append((int(end) - int(start)) / 10e3)

    
x=np.asarray(x)
y=np.asarray(y)
z=np.abs(x-y)
z=np.less(z,1).astype(np.int32)
z=np.sum(z)

loss=np.mean(np.square(x-y))
print(loss)
print(z/y.shape[0])