from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from config import *

os.chdir(WORKING_DIR)

use_cuda = torch.cuda.is_available()
print("GPU availability is:", use_cuda)

SOS_token = 0
EOS_token = 1
hidden_size = HIDDEN_SIZE
teacher_forcing_ratio = TEACHER_FORCING_RATIO

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/processed/{}-{}_{}-{}.txt'.format(EVAL_TRNorTST, TASK_NAME, lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse=False)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('in', 'out', True)


if EMBEDDEING_SOURCE == 'google':
    with open('data/emb_pretrained/embedding_GoogleNews300Negative.pkl'.format(hidden_size), 'rb') as handle:
        b = pickle.load(handle)
else:
    with open('data/emb_pretrained/embedding_raw{}d.pkl'.format(hidden_size), 'rb') as handle:
        b = pickle.load(handle)

pretrained_emb = np.zeros((input_lang.n_words, hidden_size))
for k, v in input_lang.index2word.items():
    if v == 'SOS':
        pretrained_emb[k] = np.zeros(hidden_size)
    elif (v == 'EOS') and (EMBEDDEING_SOURCE != 'google'):
        pretrained_emb[k] = b['.']
    elif (v == 'and') and (EMBEDDEING_SOURCE == 'google'):
        pretrained_emb[k] = b['AND']
    else:
        pretrained_emb[k] = b[v]



## Language Model
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained_emb, model_type='GRU'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        if EMBEDDEING_PRETRAINED:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.embedding.weight.requires_grad = WEIGHT_UPDATE

        self.gru = getattr(nn, model_type)(hidden_size, hidden_size, bidirectional=False)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)) )

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    if os.path.exists("saved_models/encoder_" + MODEL_VERSION):
        encoder = torch.load("saved_models/encoder_" + MODEL_VERSION)
        decoder = torch.load("saved_models/decoder_" + MODEL_VERSION)
        
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



def evaluateAccuracy(encoder, decoder, n=10):
    ACCs = []
    for i in range(n):
        pair = random.choice(pairs_eval)
        output_words, _ = evaluate(encoder, decoder, pair[0])
        
        if output_words[-1] == '<EOS>':
            output_words = output_words[:-1]
        output_sentence = ' '.join(output_words)
        
        if output_sentence == pair[1]:
            ACCs.append(1)
        else:
            ACCs.append(0)
    return np.array(ACCs).mean()



encoder1 = EncoderRNN(input_lang.n_words, hidden_size, pretrained_emb)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)


if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()


print("Calculating accuracy on testing data. . .")
test_acc = evaluateAccuracy(encoder1, attn_decoder1, n=5000)
print("Testing accuracy =", test_acc)

evaluateRandomly(encoder=encoder1, decoder=attn_decoder1, n=10)


