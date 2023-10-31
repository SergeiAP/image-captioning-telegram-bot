import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data.config import image_captioning
from .vocab import VOCAB_SIZE, PAD_IDX, EOS_IDX, SOS_IDX


class CaptionNet(nn.Module):
    def __init__(self, n_tokens, max_len, device, emb_dim=128, lstm_units=256, cnn_feature_size=2048,
                 dropout=0.2, bidirectional=False, batch_first=True, num_layers=2):
        super(self.__class__, self).__init__()

        self.n_tokens = n_tokens
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.lstm_units = lstm_units
        self.cnn_feature_size = cnn_feature_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        # два линейных слоя, которые будут из векторов, полученных на выходе Inseption,
        # получать начальные состояния h0 и c0 LSTM-ки, которую мы потом будем
        # разворачивать во времени и генерить ею текст
        self.cnn_to_h0 = nn.Linear(self.cnn_feature_size, self.lstm_units)
        self.cnn_to_c0 = nn.Linear(self.cnn_feature_size, self.lstm_units)

        # вот теперь recurrent part

        # create embedding for input words. Use the parameters (e.g. emb_dim).
        self.embedding = nn.Embedding(num_embeddings=self.n_tokens,
                                      embedding_dim=self.emb_dim)

        # lstm: настакайте LSTM-ок (1 или более, но не надо сразу пихать больше двух, замучаетесь ждать).
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.lstm_units,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=batch_first
        )

        # ну и линейный слой для получения логитов
        self.out = nn.Linear(
            in_features=(1 + self.bidirectional) * self.lstm_units,
            out_features=self.n_tokens

        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, image_vectors, captions_ix, teacher_forcing_ratio, topk=1):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor, содержаший выходы inseption. Те, из которых будем генерить текст
                shape: (batch_size x cnn_feature_size)
        :param captions_ix: таргет описания картинок в виде матрицы
                shape: (batch_size x max_len)
        :returns: логиты для сгенерированного текста описания
                shape: (batch_size x max_len x n_tokens)
        """
        teacher_forcing_ratio = teacher_forcing_ratio[0]
        batch_size = image_vectors.shape[0]

        cell = self.cnn_to_c0(image_vectors).unsqueeze(0).repeat((1 + self.bidirectional) * self.num_layers, 1,
                                                                 1)  # (batch_size x cnn_feature_size) -> (num_directions*num_layers x batch_size x hid_dim)
        hidden = self.cnn_to_h0(image_vectors).unsqueeze(0).repeat((1 + self.bidirectional) * self.num_layers, 1,
                                                                   1)  # (batch_size x cnn_feature_size) -> (num_directions*num_layers x batch_size x hid_dim)

        # применим LSTM:
        # 1. инициализируем lstm state с помощью initial_* (сверху)
        # 2. скормим LSTM captions_emb
        # 3. посчитаем логиты из выхода LSTM
        if teacher_forcing_ratio == 1:

            # first input to the decoder is the <sos> tokens
            input_ = captions_ix[:, :-1]
            captions_emb = self.embedding(input_)
            captions_emb = self.dropout(captions_emb)
            lstm_out, (hidden, cell) = self.rnn(captions_emb, (hidden, cell))
            logits = self.out(lstm_out)

        else:
            # first input to the decoder is the <sos> tokens
            logits = torch.zeros(batch_size, self.max_len - 1, self.n_tokens).to(
                self.device)  # to store logits (batch_size x max_len x n_tokens)
            input_ = torch.ones((batch_size,), dtype=torch.int64).to(self.device) * SOS_IDX  # [batch_size]

            for t in range(1, self.max_len):

                input_ = input_.unsqueeze(1)  # -> (batch_size x 1)
                captions_emb = self.embedding(input_)  # -> (batch_size x 1 x hid_dim)
                captions_emb = self.dropout(captions_emb)

                # output -> (batch_size x 1 x num_directions*hid_dim)
                # hidden -> (num_directions*num_layers x batch_size x hid_dim)
                # cell -> (num_directions*num_layers x batch_size x hid_dim)
                lstm_out, (hidden, cell) = self.rnn(captions_emb, (hidden, cell))

                logit = self.out(lstm_out.squeeze(1))  # -> (batch_size x n_tokens)

                logits[:, t - 1, :] = logit  # (batch_size x max_len x n_tokens)
                teacher_force = random.random() < teacher_forcing_ratio
                if topk == 1:
                    top1 = logit.max(1)[1]  # -> (batch_size)
                else:
                    topk_logit, topk_idxs = torch.topk(logit, k=topk, dim=1)  # -> (batch_size x beam_szie)
                    # top1 = topk_idxs[:,torch.randint(0, topk, (1,))] # random sample version

                    topk_logit = F.softmax(topk_logit, dim=1).cpu().numpy()  # -> (batch_size x beam_szie)
                    topk_idxs = topk_idxs.cpu()

                    top1 = torch.tensor(
                        [np.random.choice(topk_idxs[idx], 1, p=topk_logit[idx]) for idx in range(batch_size)],
                        dtype=torch.int64)  # weighted sample version

                    top1 = top1.squeeze(1).to(self.device)
                input_ = (captions_ix[:, t] if teacher_force else top1)

        return logits  # -> (batch_size x max_len-1 x n_tokens)

    def test(self, image_vectors, topk=1):
        # for using model in eval for pictures
        """
        :returns top1: chosen words (batch_size x max_len-1)
        :returns probs: sentence probability - (max_len-1) root of multiplication of all predictied words in sentence - like distance (batch_size)
        """
        with torch.no_grad():
            logits = self.forward(image_vectors, None, teacher_forcing_ratio=[0.0],
                                  topk=topk)  # -> (batch_size x max_len-1 x n_tokens)
            logits = F.softmax(logits, dim=2)
            batch_size, max_len = logits.shape[:2]
            if topk == 1:
                probs, top1 = logits.max(2)
                probs = torch.pow(torch.prod(probs, dim=1), 1 / max_len)

            else:
                top1 = []
                topk_logits, topk_idxs = torch.topk(logits, k=topk, dim=2)  # -> (batch_size x max_len x beam_szie)
                # top1 = topk_logits[:,:,torch.randint(0, topk, (1,))].squeeze(2) # -> (batch_size x max_len) - random sample version
                topk_logits = F.softmax(topk_logits, dim=2).cpu().numpy()  # -> (batch_size x max_len x beam_szie)
                topk_idxs = topk_idxs.cpu()
                for i_bs in range(batch_size):
                    top1.append([np.random.choice(topk_idxs[i_bs, i_ml], 1, p=topk_logits[i_bs, i_ml]) for i_ml in
                                 range(max_len - 1)])  # weighted sample version
                top1 = torch.tensor(top1, dtype=torch.int64).to(self.device)
                probs = torch.gather(logits[:, :-1, :], dim=2, index=top1)
                top1 = top1.squeeze(2)
                probs = torch.pow(torch.prod(probs.squeeze(2), dim=1), 1 / (max_len - 1)).cpu().numpy()
        return top1, probs  # -> (batch_size x max_len) and (batch_size)

def fix_seed():
    seed = image_captioning['general']['SEED']
    torch.manual_seed(image_captioning['general']['SEED'])

fix_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_model_1_layer = CaptionNet(device=DEVICE,
                        n_tokens=VOCAB_SIZE,
                        max_len=image_captioning['general']['MAX_LEN'],
                        emb_dim=image_captioning['model_parameters']['1_layer_simple_LSTM']['EMB_DIM'],
                        lstm_units=image_captioning['model_parameters']['1_layer_simple_LSTM']['HIDDEN_SIZE'],
                        cnn_feature_size=image_captioning['model_parameters']['1_layer_simple_LSTM']['CNN_FEATURE_SIZE'],
                        dropout=image_captioning['model_parameters']['1_layer_simple_LSTM']['DROPOUT'],
                        bidirectional=image_captioning['model_parameters']['1_layer_simple_LSTM']['BIDIRECTIONAL'],
                        batch_first=True,
                        num_layers=image_captioning['model_parameters']['1_layer_simple_LSTM']['NUM_LAYERS'])

LSTM_model_1_layer = LSTM_model_1_layer.to(DEVICE)

LSTM_model_2_layer = CaptionNet(device=DEVICE,
                        n_tokens=VOCAB_SIZE,
                        max_len=image_captioning['general']['MAX_LEN'],
                        emb_dim=image_captioning['model_parameters']['2_layer_simple_LSTM']['EMB_DIM'],
                        lstm_units=image_captioning['model_parameters']['2_layer_simple_LSTM']['HIDDEN_SIZE'],
                        cnn_feature_size=image_captioning['model_parameters']['2_layer_simple_LSTM']['CNN_FEATURE_SIZE'],
                        dropout=image_captioning['model_parameters']['2_layer_simple_LSTM']['DROPOUT'],
                        bidirectional=image_captioning['model_parameters']['2_layer_simple_LSTM']['BIDIRECTIONAL'],
                        batch_first=True,
                        num_layers=image_captioning['model_parameters']['2_layer_simple_LSTM']['NUM_LAYERS'])

LSTM_model_2_layer = LSTM_model_2_layer.to(DEVICE)