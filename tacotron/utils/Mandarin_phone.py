import os

phone_dict={}
with open(os.path.join(os.path.expandvars('$HOME'),'my_tacotron2/tacotron/utils/Mandarin_phone.txt')) as reader:
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
