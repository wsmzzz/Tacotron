import os

phone_dict={}
with open('/data5/wangshiming/my_tacotron2/tacotron/utils/Mandarin_phone.txt','r') as reader:
    lines=reader.readlines()
    for index,i in enumerate(lines):
        phone_dict[i.strip()]=index
      
   
"""
#1 = #
#2 = *
#3 = ^
#4 = &
"""
   
rhythm_dict={
    'pad':0,
    '#':1,
    '*':2,
    '^':3,
    '&':4,
}

print(len(phone_dict))
