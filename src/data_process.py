import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("../data/raw/tasks_train_addprim_jump.txt", "r", encoding="utf8") as f:
    
    IN_seq = []
    OUT_seq = []
    
    for i, line in enumerate(f):
        line = line.split(' OUT: ')
        
        IN_seq.append(line[0][4:].strip())
        OUT_seq.append(line[1].strip())

def get_voc(SEQ):
    voc = []
    seq_len = []
    for seq in SEQ:
        seq_len.append(len(seq))
        wd_list = seq.split(' ')
        for wd in wd_list:
            if wd not in voc:
                voc.append(wd)
    return voc, seq_len


voc_in, inSeq_len = get_voc(IN_seq)
voc_out, outSeq_len = get_voc(OUT_seq)


with open("../data/processed/train-addprim-jump_in-out.txt", "a", encoding="utf8") as f:
    
    for i, j in zip(IN_seq, OUT_seq):
        f.write(i+'\t'+j+'\n')