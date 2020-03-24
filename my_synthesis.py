import argparse
import os
from warnings import warn
from time import sleep
import tensorflow as tf
from hparams import hparams
from tacotron.feeder import get_dur,batch_convert_dur2alignment
from infolog import log
from tacotron.synthesizer import Synthesizer
import tqdm
import  numpy as np

def pad_1D(pad_list):
    length=[len(i) for i in pad_list]
    max_len=max(length)
    pad=[np.pad(i,pad_width=[0,max_len-len(i)],mode='constant',constant_values=0)  for i in pad_list]
    return np.stack(pad,axis=0)


if __name__=='__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    argparser=argparse.ArgumentParser()
    argparser.add_argument('--base_dir',default='Data')
    argparser.add_argument('--input',default='training_data/test.txt')
    argparser.add_argument('--batch_size',default=20)
    argparser.add_argument('--out_dir', default='mask_Synthesizer')
    argparser.add_argument('--a',default='0')
    argparser.add_argument('--dur_inputs_dir',default='/data5/wangshiming/biaobei/biaobei/lab')
    argparser.add_argument('--checkpoint',default='/data5/wangshiming/my_tacotron2/train_log/logs_a=0-Tacotron/taco_pretrained/tacotron_model.ckpt-146000')
    args=argparser.parse_args()
    out_dir=os.path.join(args.out_dir,'a=0 dur')


    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)

    with open(os.path.join(args.base_dir,args.input),mode='r') as reader:
        lines=reader.readlines()
        name= [line.strip().split("|")[0].rstrip('.npy').lstrip('audio-') for line in lines]
        texts=[line.strip().split("|")[5] for line in lines]
    name = np.reshape(np.asarray(name), [-1, args.batch_size])
    syther=Synthesizer()
    syther.load(checkpoint_path=args.checkpoint,hparams=hparams,a=args.a)

   
    for i in tqdm.tqdm(range(name.shape[0])):
        dur=[]
        for lab_name in name[i, :]:
        #     lad_dir=os.path.join(args.dur_inputs_dir,'%06d.label'%(int(lab_name)))
        #     dur.append(get_dur(lad_dir))
        # dur=pad_1D(dur)
        # alignment_merlin=batch_convert_dur2alignment(dur)

        syther.my_synthesize(texts[i * args.batch_size:(i + 1) * args.batch_size], basenames=name[i, :],
                                  out_dir=out_dir)



    




