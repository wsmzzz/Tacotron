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
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--base_dir',default='Data')
    argparser.add_argument('--input',default='training_data/test.txt')
    argparser.add_argument('--batch_size',default=2)
    argparser.add_argument('--out_dir', default='Synthesizer_train')
    argparser.add_argument('--checkpoint',default='/data5/wangshiming/my_tacotron2/train_log_dur/logs-Tacotron/taco_pretrained/tacotron_model.ckpt-13000')
    args=argparser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.base_dir,args.input),mode='r') as reader:
        lines=reader.readlines()
        name= [line.strip().split("|")[0].rstrip('.npy').lstrip('audio-') for line in lines]
        texts=[line.strip().split("|")[5] for line in lines]
    syther=Synthesizer()
    syther.load(checkpoint_path=args.checkpoint,hparams=hparams)
    name=np.reshape(np.asarray(name),[-1 ,args.batch_size])
   
    for i in range(name.shape[0]):
        # syther.my_synthesize(texts[i*args.batch_size:(i+1)*args.batch_size],basenames=name[i,:],out_dir=args.out_dir)
        syther.get_dur(texts[i * args.batch_size:(i + 1) * args.batch_size], basenames=name[i, :],
                             out_dir=args.out_dir)
    




