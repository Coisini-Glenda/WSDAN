from sentence_transformers import SentenceTransformer
import random
from transformers import BertTokenizer, BertModel
import os
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import BertModel, AutoModel
from nltk.translate.bleu_score import sentence_bleu

from torchvision import models, transforms
from torchvision.io import read_image, image
import timm
import torch.nn.functional as F
import warnings
import torch, math
import torch.nn as nn
import numpy as np
import json
warnings.simplefilter("ignore", UserWarning)

sen = 'pre-model/sentence-transformers_all-mpnet-base-v2'
bert_base = 'pre-model/bert-base-uncased'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def onehot(size, target):

    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_data(args):
    train_file = open(os.path.join(args.data_dir, 'trainset.json'), )
    test_file = open(os.path.join(args.data_dir, 'testset.json'), )

    train_data = json.load(train_file)
    test_data = json.load(test_file)

    traindf = pd.DataFrame(train_data)
    traindf['mode'] = 'train'
    testdf = pd.DataFrame(test_data)
    testdf['mode'] = 'test'

    traindf['image_name'] = traindf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    testdf['image_name'] = testdf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))

    traindf['question_type'] = traindf['question_type'].str.lower()
    testdf['question_type'] = testdf['question_type'].str.lower()

    return traindf, testdf

def encode_text(caption, tokenizer, args):


    word_piece = tokenizer.tokenize(caption) 
    part = tokenizer.convert_tokens_to_ids(word_piece)  

    tokens = part[:args.max_seq_len]
    input_text_mask = [1] * len(tokens)  
    n_pad = args.max_seq_len - len(tokens)
    tokens.extend([0] * n_pad)  
    input_text_mask.extend([0] * n_pad)  
    input_img_mask = [1] * 5 
    return tokens, input_text_mask, input_img_mask


def calculate_bleu_score(preds, targets, idx2ans):
    bleu_per_answer = np.asarray(
        [sentence_bleu([idx2ans[target].split()], idx2ans[pred].split(), weights=[1]) for pred, target in
         zip(preds, targets)])
    return np.mean(bleu_per_answer)


class VQARAD(Dataset):
    def __init__(self, df, args, mode='Train'):
        super().__init__()
        self.df = df.values
        self.args = args
        self.pretrained = BertTokenizer.from_pretrained(bert_base)
        self.tokenizer = self.pretrained

        # self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)  # wordpiece分词?
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df[index, 1] 
        question = self.df[index, 6]
        answer = self.df[index, 3]

        tokens, question_mask, img_mask = encode_text(question,self.tokenizer, self.args)
        if self.args.smoothing:
            answer = onehot(self.args.num_classes, answer)
            
        return path, question, torch.tensor(tokens, dtype=torch.long), torch.tensor(question_mask,dtype=torch.float32), torch.tensor(
            img_mask, dtype=torch.float32), torch.tensor(answer, dtype=torch.long)

    
class Transfer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = models.resnet152(pretrained=True).to(device)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1, 1))
        self.tfm = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.95, 1.05), ratio=(0.95, 1.05)),
             transforms.RandomRotation(10),
             ZeroOneNormalize(),
             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def forward(self, path):
        path = list(path)
        img_list = []
        for ph in path:
            img = read_image(ph, image.ImageReadMode.RGB)
            # img = img.to(device)
            img = self.tfm(img)
            img_list.append(img)
        img = torch.stack(img_list).to(device)

        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        inter_2 = self.conv2(fix2(img))
        v_2 = self.gap2(self.relu(inter_2)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        inter_3 = self.conv3(fix3(img))
        v_3 = self.gap3(self.relu(inter_3)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        inter_4 = self.conv4(fix4(img))
        v_4 = self.gap4(self.relu(inter_4)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        inter_5 = self.conv5(fix5(img))
        v_5 = self.gap5(self.relu(inter_5)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        inter_7 = self.conv7(fix7(img))
        v_7 = self.gap7(self.relu(inter_7)).view(-1, self.args.hidden_size).unsqueeze(1)
     
        return torch.cat((v_2, v_3, v_4, v_5, v_7), 1) 


class word_sen(nn.Module):
    def __init__(self,args):
        super(word_sen,self).__init__()
        self.args = args

        self.proj_word_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_word_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_word_v = nn.Linear(args.hidden_size, args.hidden_size)

        self.proj_sen_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_sen_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_sen_v = nn.Linear(args.hidden_size, args.hidden_size)

        # self.proj_pos_q = nn.Linear(args.hidden_size, args.hidden_size)
        # self.proj_pos_k = nn.Linear(args.hidden_size, args.hidden_size)
        # self.proj_pos_v = nn.Linear(args.hidden_size, args.hidden_size)

        self.drop = nn.Dropout(args.dropout)
        self.scores = None
        self.n_heads = args.num_heads

        # self.P=PositionalEncoding(args)

    def forward(self, word, sen, mask):

        q, k, v = self.proj_word_q(word), self.proj_word_k(word), self.proj_word_v(word)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        sen_q, sen_k = self.proj_sen_q(sen), self.proj_sen_k(sen)
        sen_q, sen_k = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [sen_q, sen_k])


        scores = q @ k.transpose(-2, -1) / np.sqrt(2 * k.size(-1)) + sen_q @ sen_k.transpose(-2, -1) / np.sqrt(2 * sen_k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h

    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)


class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        base_model = BertModel.from_pretrained(bert_base)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0] 

        self.word_embedding = nn.Linear(768, self.args.hidden_size, bias=False).to(device)
        nn.init.xavier_normal_(self.word_embedding.weight)
        self.sen_encode = SentenceTransformer(sen)
        self.sen_embedding = nn.Linear(768, self.args.hidden_size, bias=False).to(device)  
        nn.init.xavier_normal_(self.sen_embedding.weight)
        self.w_s = word_sen(args)

    def forward(self, input_tokens, input_sen, question_mask):
        word_embedding = self.bert_embedding(input_tokens)
        tokens_embedding = self.word_embedding(word_embedding)

        sen = self.sen_encode.encode(input_sen)
        sen = torch.tensor(sen)
        sen = sen.unsqueeze(1).to(device)
        sen_embedding = self.sen_embedding(sen)

        embeddings = self.w_s(tokens_embedding, sen_embedding, question_mask)

        return embeddings 

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, args):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.fc2 = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.dropout)
        self.scores = None
        self.n_heads = args.num_heads
    def forward(self, q, k, v, mask):
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class CTDN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.dropx1 = nn.Dropout(args.dropout)
        self.dropy1 = nn.Dropout(args.dropout)
        self.dropx2 = nn.Dropout(args.dropout)
        self.dropy2 = nn.Dropout(args.dropout)
        self.dropx3 = nn.Dropout(args.dropout)
        self.dropy3 = nn.Dropout(args.dropout)

        self.attention_x = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
        self.normx1 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.co_atten_x = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
        self.normx2 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.FFN_cox = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        self.normx3 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.attention_y = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
        self.normy1 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.co_atten_y = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
        self.normy2 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.FFN_coy = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        self.normy3 = nn.LayerNorm(args.hidden_size, eps=1e-12)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, x_mask, y_mask, layer_num):

        x = self.normx1(x + self.dropx1(self.attention_x[layer_num](x, x, x, x_mask)))
        y = self.normy1(y + self.dropy1(self.attention_y[layer_num](y, y, y, y_mask)))

        x = self.normx2(x + self.dropx2(self.co_atten_x[layer_num](x, y, y, y_mask)))
        y = self.normy2(y + self.dropy2(self.co_atten_y[layer_num](y, x, x, x_mask)))

        x = self.normx3(x + self.dropx3(self.FFN_cox[layer_num](x)))
        y = self.normy3(y + self.dropy3(self.FFN_coy[layer_num](y)))
        return x, y

class MCTDN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.blocks = CTDN(args)
        self.n_layers = args.n_layers
        self.question_embedding = Embedding(args).to(device)
        self.visual_embedding = Transfer(args).to(device)
        # self.LayNorm = nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True)

    def forward(self, img, sens, tokens, question_mask, img_mask):
        hy = self.question_embedding(tokens, sens, question_mask)
        hx = self.visual_embedding(img)
        for i in range(self.n_layers):
            hx, hy = self.blocks(hx, hy, img_mask, question_mask, i)

        return torch.cat([hx, hy], dim=1)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mcade = MCTDN(args).to(device)
        self.fusion = nn.Linear(25, 1).to(device)

        self.classifer = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                       nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                       nn.Linear(args.hidden_size, 1749)).to(device)

    def forward(self, img, sens, tokens, question_mask, img_mask):
        h = self.mcade(img, sens, tokens, question_mask, img_mask)  
        h = h.permute(0, 2, 1)
        pooled_h = self.fusion(h).squeeze(2)
        logits = self.classifer(pooled_h)
        return logits


def train_one_epoch(loader, model, optimizer, criterion, scaler, idx2ans, args, train_df):
    model.train()
    Target = []
    Preds = []
    count, loss_sum = 0, 0
    loss_func = criterion
    bar = tqdm(loader, leave=False)

    for (img, sens, tokens, question_mask, img_mask, answer) in bar:
        tokens, question_mask, img_mask, answer = tokens.to(device), question_mask.to(device), img_mask.to(device), answer.to(device)
        count += 1

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(img, sens, tokens, question_mask, img_mask)
            loss = loss_func(logits, answer)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = logits.softmax(1).argmax(1).detach()
        Preds.append(pred)

        loss_sum += loss.item()

        bar.set_description('train_loss: %.5f' % loss.item())
        if args.smoothing:
            Target.append(answer.argmax(1))
        else:
            Target.append(answer)
    Preds = torch.cat(Preds).cpu().numpy()
    Target = torch.cat(Target).cpu().numpy()

    total_acc = (Preds==Target).mean()*100.
    return loss_sum / count, total_acc


def test(loader, model, criterion, args, test_df, idx2ans):
    model.eval()
    Preds = []
    Targets = []
    loss_sum, count = 0, 0

    with torch.no_grad():
        for (img, sens, tokens, question_mask, img_mask, answer) in loader:

            count += 1
            tokens, question_mask, img_mask, answer = tokens.to(device), question_mask.to(device), img_mask.to(
                device), answer.to(device)
            with torch.cuda.amp.autocast():
                logits = model(img,sens, tokens, question_mask, img_mask)
                # logits = model(img, tokens, question_mask, img_mask)
                loss = criterion(logits, answer)

            loss = loss.mean()
            loss_sum += loss.item()

            pred = logits.softmax(1).argmax(1).detach()
            Preds.append(pred)
            if args.smoothing:
                Targets.append(answer.argmax(1))
            else:
                Targets.append(answer)

        test_loss = loss_sum / count

    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()

    total_acc = (Preds == Targets).mean() * 100.
    return test_loss, total_acc
