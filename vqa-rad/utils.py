import random
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from torchvision import models, transforms
import torch.nn.functional as F
import warnings
import albumentations as A
import torch, math
import torch.nn as nn
import numpy as np
from einops import rearrange
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
# from eval import *
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import label_binarize, OneHotEncoder
import matplotlib.pyplot as plt


warnings.simplefilter("ignore", UserWarning)

bert_base = '../pre-model/bert-base-uncased'
sen = '../pre-model/sentence-transformers_all-mpnet-base-v2'
biobert = '../pre-model/biobert-base-cased-v1.1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def calculate_bleu_score(preds, targets, idx2ans):
    bleu_per_answer = np.asarray(
        [sentence_bleu([str(idx2ans[target]).split()], str(idx2ans[pred]).split(), weights=[1]) for pred, target in
         zip(preds, targets)])
    return np.mean(bleu_per_answer)


def onehot(size, target):
    '''
    将答案转化为 one-hot 编码
    '''
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
    '''
    读取文件
    '''
    traindf = pd.read_csv(os.path.join(args.data_dir, 'train.txt'), sep='|',
                          names=['img_id', 'question', 'answer', 'category', 'mode'])
    validf = pd.read_csv(os.path.join(args.data_dir, 'val.txt'), sep='|',
                         names=['img_id', 'question', 'answer', 'category', 'mode'])
    testdf = pd.read_csv(os.path.join(args.data_dir, 'test.txt'), sep='|',
                         names=['img_id', 'question', 'answer', 'category', 'mode'])

    # remove_train2019 = ['synpic21456', 'synpic21845', 'synpic47995', 'synpic48869', 'synpic52613', 'synpic31716',
    #                     'synpic27917', 'synpic39365', 'synpic19434', 'synpic52600',
    #                     'synpic56649', 'synpic52603', 'synpic52610', 'synpic46659', 'synpic19533']
    # traindf = traindf[~traindf['img_id'].isin(remove_train2019)].reset_index(drop=True)
    '''
    将读取的文件图片id与iamges文件中的图片对应起来
    '''
    traindf['img_id'] = traindf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'Train', x + '.npy'))
    validf['img_id'] = validf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'Valid', x + '.npy'))
    testdf['img_id'] = testdf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'Test', x + '.npy'))

    return traindf, validf, testdf


def encode_text(caption, tokenizer, args):
    '''
    这里tokenizer使用wordpiece
    '''

    word_piece = tokenizer.tokenize(caption)  # 使用wordpiece得出的分词
    part = tokenizer.convert_tokens_to_ids(word_piece)  # 将对应的分词转换为索引

    tokens = part[:args.max_seq_len]
    input_text_mask = [1] * len(tokens)  # 有内容的mask规定为1
    n_pad = args.max_seq_len - len(tokens)
    tokens.extend([0] * n_pad)  # 对没有内容的位置进行0填充
    input_text_mask.extend([0] * n_pad)  # 填充的内容mask为0
    input_img_mask = [1] * 5  # 将图像的mask全部归为1，确保可以进行交互
    return tokens, input_text_mask, input_img_mask


class VQAMed2019(Dataset):
    def __init__(self, df, args):
        super().__init__()
        self.df = df
        self.args = args
        # self.tokenizer = BertTokenizer.from_pretrained(bert_base)
        self.tokenizer = AutoTokenizer.from_pretrained("../pre-model/biobert_v1.1")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None,
                #                    mask_value=None, p=1),
                A.ColorJitter(),
                A.HorizontalFlip(),  # 水平翻转
                A.VerticalFlip()  # 垂直翻转
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = np.load(self.df.loc[index, 'img_id'])  # 根据索引index，得到img_id的位置path
        question = self.df.loc[index, 'question']
        answer = self.df.loc[index, 'answer']

        transformed = self.transform(image=img)
        image = self.img_transform(transformed['image'])
        tokens, question_mask, img_mask = encode_text(question, self.tokenizer, self.args)
        if self.args.smoothing:
            answer = onehot(self.args.num_classes, answer)
        '''
        输出的：
        img的维度为：[batch_size, 3, 224, 224]
        question的维度为：[batch_size, 4800]
        tokens的维度为：[batch_size, max_seq_len]
        question_mask的维度为：[batch_size, max_seq_len ]
        img_mask的维度为：[batch_size, 5]
        answer的维度为：[batch_size]
        '''
        # index 输出类似: tensor([1046, 1974,  641, 1289])
        # return img, question, answer
        # return path, torch.tensor(tokens, dtype=torch.long), torch.tensor(question_mask,dtype=torch.float32), torch.tensor(img_mask, dtype=torch.float32), torch.tensor(answer, dtype=torch.long)
        return image, question, torch.tensor(tokens, dtype=torch.long), torch.tensor(question_mask,
                                                                                     dtype=torch.float32), torch.tensor(
            img_mask, dtype=torch.float32), torch.tensor(answer, dtype=torch.long)


class Transfer_grad(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = models.resnet152(pretrained=True).to(device)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv7 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1, 1))
        self.grad_cam = False

    def forward(self, img):
        img = img.to(device)  # torch.Size([16, 3, 224, 224])
        modules2 = list(self.model.children())[:-7]
        fix2 = nn.Sequential(*modules2)
        inter_2 = self.conv2(fix2(img))
        v_2 = self.gap2(self.relu(inter_2)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules3 = list(self.model.children())[:-5]
        fix3 = nn.Sequential(*modules3)
        inter_3 = self.conv3(fix3(img))
        v_3 = self.gap3(self.relu(inter_3)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        inter_4 = self.conv4(fix4(img))
        v_4 = self.gap4(self.relu(inter_4)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules5 = list(self.model.children())[:-3]
        fix5 = nn.Sequential(*modules5)
        inter_5 = self.conv5(fix5(img))
        v_5 = self.gap5(self.relu(inter_5)).view(-1, self.args.hidden_size).unsqueeze(1)
        modules7 = list(self.model.children())[:-2]
        fix7 = nn.Sequential(*modules7)  # torch.Size([16, 312, 112, 112])
        o_7 = fix7(img)  # torch.size([16,64,112,112])

        if self.grad_cam:
            self.feat = o_7
        inter_7 = self.conv7(o_7)
        v_7 = self.gap7(self.relu(inter_7)).view(-1, self.args.hidden_size).unsqueeze(1)
        return torch.cat((v_2, v_3, v_4, v_5, v_7), 1)  # 维度为 [batch_size, 5, hidden_size]

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.feat


class TSE(nn.Module):
    def __init__(self, args):
        super().__init__()

        base_model = BertModel.from_pretrained(bert_base)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]  # 输出维度为 [batch_size, max_seq_len, 768]
        self.word_embedding = nn.Linear(768, args.hidden_size, bias=False).to(device)  # 将单词嵌入转为为hidden_size

        self.sen = AutoModel.from_pretrained("../pre-model/biobert_v1.1")
        self.sen_embedding = nn.Linear(768, args.hidden_size, bias=False).to(device)

        # self.sen_encode = SentenceTransformer(sen)
        # self.sen_embedding = nn.Linear(768, args.hidden_size, bias=False).to(device)  # 将句子嵌入表示转为hidden_size

        # self.word_embeddings = nn.Embedding(args.vocab_size, 768, padding_idx=0)
        # self.word_embedding = nn.Linear(768, args.hidden_size, bias=False)

        self.heads = args.num_heads

        self.scale = (2 * args.head_dim) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)

        self.to_qkv1 = nn.Linear(args.hidden_size, args.hidden_size * 3, bias=False)
        self.to_qkv2 = nn.Linear(args.hidden_size, args.hidden_size * 2, bias=False)

    def forward(self, w, s, mask):
        # word_embedding = self.word_embeddings(w)
        # tokens_embedding = self.word_embedding(word_embedding)
        word_embedding = self.bert_embedding(w)
        tokens_embedding = self.word_embedding(word_embedding)

        sen_embedding = self.sen(w)
        sen_embedding = self.sen_embedding(sen_embedding.pooler_output.unsqueeze(1))

        # sen = self.sen_encode.encode(s)
        # sen = torch.tensor(sen)
        # sen = sen.unsqueeze(1).to(device)
        # sen_embedding = self.sen_embedding(sen)

        qkv1 = self.to_qkv1(tokens_embedding).chunk(3, dim=-1)
        qkv2 = self.to_qkv2(sen_embedding).chunk(2, dim=-1)

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv1)
        q2, k2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv2)

        dots = self.scale * (torch.matmul(q1, k1.transpose(-1, -2)) + torch.matmul(q2, k2.transpose(-1, -2)))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            dots -= 10000.0 * (1.0 - mask)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.hidden_size, 2 * args.hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(2 * args.hidden_size, args.hidden_size),
            nn.Dropout(args.dropout),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)
        nn.init.normal_(self.net[0].bias, std=1e-6)
        nn.init.normal_(self.net[3].bias, std=1e-6)

    def forward(self, x):
        return self.net(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.heads = args.num_heads
        self.scale = args.head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)

        # self.to_qkv1 = nn.Linear(args.hidden_size, args.hidden_size * 3, bias=False)
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(self, q, k, v, mask):
        qkv = (self.proj_q(q), self.proj_k(k), self.proj_v(v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        scores = self.scale * (torch.matmul(q, k.transpose(-1, -2)))

        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)

        scores = self.dropout(self.attend(scores))
        h = torch.matmul(scores, v)
        h = rearrange(h, 'b h n d -> b n (h d)')
        return h


class DAL(nn.Module):
    def __init__(self, args):
        super(DAL, self).__init__()
        '''
        DAL左边
        '''
        self.attention_left = MultiHeadedSelfAttention(args)
        self.norm = nn.LayerNorm(args.hidden_size, eps=1e-12)

        self.co_attention_left = MultiHeadedSelfAttention(args)
        self.FFN_left = FeedForward(args)

        '''
        DAL右边
        '''
        self.attention_right = MultiHeadedSelfAttention(args)
        self.co_attention_right = MultiHeadedSelfAttention(args)
        self.FFN_right = FeedForward(args)

    def forward(self, x, y, x_mask, y_mask):
        '''
        x表示 Visual；
        y表示 Question
        '''
        x = self.norm(x + self.attention_left(x, x, x, x_mask))
        y = self.norm(y + self.attention_right(y, y, y, y_mask))

        # 协同注意力计算
        x = self.norm(x + self.co_attention_left(x, y, y, y_mask))
        y = self.norm(y + self.co_attention_right(y, x, x, x_mask))

        x = self.norm(x + self.FFN_left(x))
        y = self.norm(y + self.FFN_right(y))
        return x, y


class DALs(nn.Module):
    def __init__(self, args):
        super(DALs, self).__init__()
        self.dals = nn.ModuleList([DAL(args) for _ in range(args.n_layers)])
        self.num_layers = args.n_layers

    def forward(self, x, y, x_mask, y_mask):
        for i in range(self.num_layers):
            x, y = self.dals[i](x, y, x_mask, y_mask)
        return torch.cat([x, y], dim=1)


class DALNet_WSE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.question_encoder = TSE(args)
        # self.sen_encode = SentenceTransformer(sen)
        # self.sen_embedding = nn.Linear(768, args.hidden_size, bias=False).to(device)

        # base_model = BertModel.from_pretrained(bert_base)
        # bert_model = nn.Sequential(*list(base_model.children())[0:])
        # self.bert_embedding = bert_model[0]  # 输出维度为 [batch_size, max_seq_len, 768]
        # self.word_embedding = nn.Linear(768, args.hidden_size, bias=False).to(device)  # 将单词嵌入转为为hidden_size

        # self.word_embeddings = nn.Embedding(args.vocab_size, 768, padding_idx=0)
        # self.question_encoder = nn.Linear(768, args.hidden_size, bias=False)

        # self.word_embedding = AutoModel.from_pretrained("../pre-model/biobert_v1.1")
        # self.word_embedding1 = nn.Linear(768, args.hidden_size, bias=False).to(device)

        self.img_encoder = Transfer_grad(args)

        self.DALs = DALs(args)

        # self.fusion = nn.Linear(6, 1).to(device)
        self.fusion = nn.Linear(25, 1).to(device)
        self.activ1 = nn.Tanh()
        # self.fusion = nn.Sequential(nn.Linear(25, args.hidden_size),
        #                             nn.GELU(),
        #                             nn.Dropout(args.dropout),
        #                             nn.Linear(args.hidden_size, 1),
        #                             nn.GELU())
        self.classifer = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                       nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                       nn.GELU(),
                                       nn.Linear(args.hidden_size, args.vocab_size)).to(device)

    def forward(self, img, sens, tokens, question_mask, img_mask):
        # sen = self.sen_encode.encode(sens)
        # sen = torch.tensor(sen)
        # sen = sen.unsqueeze(1).to(device)
        # question_embedding = self.sen_embedding(sen)

        question_embedding = self.question_encoder(tokens, sens, question_mask)

        # tokens_embedding = self.word_embedding(tokens)
        # question_embedding = self.word_embedding1(tokens_embedding.last_hidden_state)

        # word_embedding = self.bert_embedding(tokens)
        # question_embedding = self.word_embedding(word_embedding)
        # tokens = self.word_embeddings(tokens)
        # question_embedding = self.question_encoder(tokens)

        visual_embedding = self.img_encoder(img)

        z = self.DALs(visual_embedding, question_embedding, img_mask,question_mask)

        # z = torch.cat([visual_embedding, question_embedding], dim=1).permute(0,2,1)
        # pooled_h = self.concat(z).squeeze(2)

        h = z.permute(0, 2, 1)  # torch.Size([16, 312, 25])
        pooled_h = self.activ1(self.fusion(h)).squeeze(2)

        logits = self.classifer(pooled_h)
        return logits


def train_one_epoch(loader, model, optimizer, criterion, scaler, scheduler, idx2ans, args):
    model.train()
    Target = []
    Preds = []
    count, loss_sum = 0, 0
    loss_func = criterion
    # bar = tqdm(loader, leave=False, ncols=85,position=0)

    with tqdm(total=len(loader), leave=False) as bar:
        for img, sens, tokens, question_mask, img_mask, answer in loader:
            img, tokens, question_mask, img_mask, answer = img.to(device), tokens.to(device), question_mask.to(
                device), img_mask.to(device), answer.to(device)
            count += 1

            # 使用混合精度
            with torch.cuda.amp.autocast():
                logits = model(img, sens, tokens, question_mask, img_mask)
                loss = loss_func(logits, answer)
                # loss = loss / args.grad_num
            scaler.scale(loss).backward()
            # if (count % args.grad_num) == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()


            pred = logits.softmax(1).argmax(1).detach()
            Preds.append(pred)

            loss_sum += loss.item()

            # bar.set_description('train_loss: %.5f' % loss.item())
            bar.update()
            if args.smoothing:
                Target.append(answer.argmax(1))
            else:
                Target.append(answer)
        Preds = torch.cat(Preds).cpu().numpy()
        Target = torch.cat(Target).cpu().numpy()

        acc = (Preds == Target).mean() * 100.
        bleu = calculate_bleu_score(Preds, Target, idx2ans)

    return loss_sum / count, acc, bleu


# 构写验证函数
def validate(loader, model, criterion, args, idx2ans):
    model.eval()
    Preds = []
    Targets = []
    loss_sum, count = 0, 0

    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, sens, tokens, question_mask, img_mask, answer) in bar:
            count += 1
            img, tokens, question_mask, img_mask, answer = img.to(device), tokens.to(device), question_mask.to(
                device), img_mask.to(
                device), answer.to(device)
            with torch.cuda.amp.autocast():
                logits = model(img, sens, tokens, question_mask, img_mask)
                loss = criterion(logits, answer)

            loss = loss.mean()
            pred = logits.softmax(1).argmax(1).detach()
            Preds.append(pred)

            if args.smoothing:
                Targets.append(answer.argmax(1))
            else:
                Targets.append(answer)

            loss_sum += loss.item()
            bar.update()

        val_loss = loss_sum / count
    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()

    acc = (Preds == Targets).mean() * 100.
    bleu = calculate_bleu_score(Preds, Targets, idx2ans)

    return val_loss, acc, bleu


# 构建测试函数
def test(loader, model, args, idx2ans):
    model.eval()
    Preds = []
    Targets = []
    pred_p =[]
    preds_p = []

    with torch.no_grad():
        for (img, sens, tokens, question_mask, img_mask, answer) in loader:

            img, tokens, question_mask, img_mask, answer = img.to(device), tokens.to(device), question_mask.to(
                device), img_mask.to(
                device), answer.to(device)
            with torch.cuda.amp.autocast():
                logits = model(img, sens, tokens, question_mask, img_mask)
            #     loss = criterion(logits, answer)
            #
            # loss = loss.mean()

            pred = logits.softmax(1).argmax(1).detach()
            # score.append(logits.softmax(1))
            pred_p.append(logits.softmax(1).cpu().numpy())
            preds_p.append(logits.softmax(1).cpu())
            Preds.append(pred)
            if args.smoothing:
                Targets.append(answer.argmax(1))
            else:
                Targets.append(answer)


    Preds = torch.cat(Preds).cpu().numpy()
    Targets = torch.cat(Targets).cpu().numpy()
    # con_mat = confusion_matrix(Targets, Preds, labels=list(idx2ans.keys()))
    if args.category == 'yesno':
        # tp, fn, fp, tn = con_mat[0,0], con_mat[0,1],con_mat[1,0],con_mat[1,1]
        acc = (Preds == Targets).mean()
        auc = roc_auc_score(Targets, Preds)
        # print('------------------------------------------------------')
        bleu = calculate_bleu_score(Preds, Targets, idx2ans)
        wbss, gt, pred = compute_wbss(Targets, Preds, idx2ans)
        return acc, bleu, auc, wbss
    else:
        y_one_hot = label_binarize(Targets, classes=np.arange(args.num_classes))
        fpr, tpr, _ = metrics.roc_curve(y_one_hot.ravel(), (np.vstack(pred_p)).ravel())
        auc = metrics.auc(fpr, tpr)


        acc = (Preds == Targets).mean()

        bleu = calculate_bleu_score(Preds, Targets, idx2ans)
        # wbss,gt, pred = compute_wbss(Targets, Preds, idx2ans)
        return acc, bleu, auc, wbss

    # recall = metrics.recall_score(Targets, Preds, average='macro')


    # return acc, bleu, sensitivity, specifivity, auc,recall
