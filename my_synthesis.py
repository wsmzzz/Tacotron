import argparse
import os
from warnings import warn
from time import sleep
import tensorflow as tf
from hparams import hparams
from infolog import log
from tacotron.synthesizer import Synthesizer
import tqdm
import  numpy as np


if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--base_dir',default='Data')
    argparser.add_argument('--input',default='training_data/test.txt')
    argparser.add_argument('--batch_size',default=2)
    argparser.add_argument('--out_dir', default='Synthesizer')
    argparser.add_argument('--a',default='1')
    argparser.add_argument('--checkpoint',default='/data5/wangshiming/my_tacotron2/train_log_dur/logs_a=1-Tacotron/taco_pretrained/tacotron_model.ckpt-32000')
    args=argparser.parse_args()
    out_dir=os.path.join(args.out_dir,args.a)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)

    with open(os.path.join(args.base_dir,args.input),mode='r') as reader:
        lines=reader.readlines()
        name= [line.strip().split("|")[0].rstrip('.npy').lstrip('audio-') for line in lines]
        texts=[line.strip().split("|")[5] for line in lines]
    syther=Synthesizer()
    syther.load(checkpoint_path=args.checkpoint,hparams=hparams,a=args.a)
    name=np.reshape(np.asarray(name),[-1 ,args.batch_size])
   
    for i in tqdm.tqdm(range(name.shape[0])):
        # syther.my_synthesize(texts[i*args.batch_size:(i+1)*args.batch_size],basenames=name[i,:],out_dir=args.out_dir)
        syther.my_synthesize(texts[i * args.batch_size:(i + 1) * args.batch_size], basenames=name[i, :],
                             out_dir=out_dir)
    




