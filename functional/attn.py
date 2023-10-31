import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vocab import VOCAB_SIZE, PAD_IDX, EOS_IDX, SOS_IDX
from data.config import image_captioning
import numpy as np

class Encoder(nn.Module):
    def __init__(self, device, src_len, kernel_size, hid_dim, num_layers, cnn_feature_size, in_channels=1,
                 bidirectional=False, dropout=0.2):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.src_len = src_len
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cnn_feature_size = cnn_feature_size
        self.bidirectional = bidirectional
        self.device = device

        self.cnn_to_h0 = nn.Linear(self.cnn_feature_size, self.hid_dim)

        # To create filters
        self.embedding = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.src_len,
            kernel_size=self.kernel_size
        )
        self.bn1 = nn.BatchNorm1d(self.src_len)
        self.dense = nn.Linear(self.cnn_feature_size - self.kernel_size + 1, (1 + self.bidirectional) * self.hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, image_vectors):
        """
        :param image_vectors: torch tensor, содержаший выходы inseption. Те, из которых будем генерить текст
                shape: (batch_size x cnn_feature_size)
        :returns:
            outputs (batch_size x src_len x emb_dim==hid_dim*num_dir)
            hidden (num_layers*num_dir x batch_size x hid_dim)
        """
        hidden = self.cnn_to_h0(image_vectors).unsqueeze(0).repeat((1 + self.bidirectional) * self.num_layers, 1, 1)
        # (batch_size x cnn_feature_size) -> (num_directions*num_layers x batch_size x cnn_feature_size)
        embedded = self.relu(self.embedding(image_vectors.unsqueeze(
            1)))  # -> (batch_size x channels x cnn_feature_size) (batch_size x src_len x emb_dim)
        # dropout over embedding
        embedded = self.bn1(self.dropout(embedded))
        outputs = self.relu(self.dense(embedded))

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim, num_layers, bidirectional=False,
                 method='concat'):  # add parameters needed for your type of attention
        super(Attention, self).__init__()

        # attention method you'll use: 'dot', 'general' or 'concat'
        self.method = method
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if self.method == 'dot':
            pass

        elif self.method == 'general':
            self.Wa = nn.Linear(
                in_features=self.hid_dim,
                out_features=self.hid_dim
            )

        elif self.method == 'concat':
            self.Wa = nn.Linear(
                in_features=self.hid_dim + self.hid_dim,
                out_features=self.hid_dim
            )
            self.va = nn.Parameter(torch.FloatTensor(self.hid_dim))

        else:
            raise NotImplementedError("Type any of methods: 'dot', 'general' or 'concat'.")

    def forward(self, last_hidden, encoder_outputs, src_len=None):
        """
        input:
          -last_hidden (batch_size x num_layers*num_dir x hid_dim)
          -encoder_outputs (batch_size x src_len x num_dir*hid_dim)
        output:
          -alignment_scores (batch_size x src_len)
        """

        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(0, 2, 1)  # -> (batch_size x num_dir*hid_dim x src_len)
            return last_hidden.bmm(encoder_outputs).squeeze(1)

        elif self.method == 'general':
            return last_hidden.bmm(self.Wa(encoder_outputs).permute(0, 2, 1)).squeeze(1)

        elif self.method == 'concat':
            src_len = encoder_outputs.shape[1]

            last_hidden = last_hidden.repeat(1, src_len, 1)  # -> (batch_size x src_len x hid_dim)

            tanh_product = torch.tanh(
                self.Wa(torch.cat((last_hidden, encoder_outputs), dim=2)))  # -> (batch_size x src_len x hid_dim)
            return tanh_product.matmul(self.va)

        else:
            raise NotImplementedError("Type any of methods: 'dot', 'general' or 'concat'.")

class LuongDecoderAttn(nn.Module):
    def __init__(self, device, n_tokens, emb_dim, hid_dim, attention, num_layers=1, bidirectional=False, dropout=0.2,
                 batch_first=True):
        super(LuongDecoderAttn, self).__init__()

        # if n_layers != 1:
        #     raise NotImplementedError('Use 1 layer only. For many layers reconstruction is required.')
        if bidirectional == True:  # model is ok for False meaning
            raise NotImplementedError('Use bidirectional=False. For other reconstruction is required.')

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_tokens = n_tokens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        # instance of Attention class
        self.attn = attention

        # define layers
        self.embedding = nn.Embedding(
            num_embeddings=self.n_tokens,
            embedding_dim=self.emb_dim
        )
        # (lstm embd, hid, layers, dropout)
        self.rnn = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=batch_first
        )

        # Projection
        self.out = nn.Linear(
            in_features=self.hid_dim + self.hid_dim,
            out_features=self.n_tokens
        )

        self.dropout = nn.Dropout(dropout)

        # more layers you'll need for attention

    def forward(self, input_, hidden, encoder_outputs, topk=1):
        """
        input:
          -input_ (batch_size)
          -hidden (num_layers*num_dir x batch_size x hid_dim)
          -encoder_outputs (batch_size x src_len x num_dir*hid_dim)
        output:
          -logit (batch_size x n_tokens)
          -hidden (num_layers*num_dir x batch_size x hid_dim)
          -attn_weights (batch_size x src_len) - for visualization
        """
        input_ = input_.unsqueeze(1)  # -> (batch_size x 1)

        embedded = self.embedding(input_)  # (batch_size x 1 x emb_dim)
        embedded = self.dropout(embedded)

        # rnn_output: (batch_size x seq_len==1 x num_directions * hidden_size)
        rnn_output, hidden = self.rnn(embedded, hidden)

        # See attention class forward for understanding
        alignment_scores = self.attn(rnn_output, encoder_outputs)  # -> (batch_size x src_len)

        # print('Alignment score: ', alignment_scores, '\n')

        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)  # (batch_size x src_len)

        # print('Weights: ', attn_weights, '\n')

        # Calculate context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size x 1 x hid_dim+hid_dim)

        logit = torch.cat((rnn_output, context), dim=-1)  # (batch_size x 1 x 2*hid_dim+hid_dim)

        # logit = F.log_softmax(self.out(logit.squeeze(1)), dim=1) # for NLLLoss

        logit = self.out(logit.squeeze(1))  # for CrossEntropyLoss

        return logit, hidden, attn_weights  # attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers * (1 + self.bidirectional), 1, self.hidden_size,
                           device=self.device)  # batch_size = 1 for decoder


class Seq2Seq(nn.Module):
    def __init__(self, device, encoder, decoder, n_tokens, max_len):
        super().__init__()
        # Hidden dimensions of encoder and decoder must be equal
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._init_weights()
        self.max_len = max_len
        self.n_tokens = n_tokens

    def forward(self, image_vectors, captions_ix, teacher_forcing_ratio, topk=1):
        """
        :param: image_vectors (batch_size x cnn_feature_size)
        :param: captions_ix (batch_size x word_i)
        :param: teacher_forcing_ratio : if 0.5 then every second token is the ground truth input
        :returns: логиты для сгенерированного текста описания shape: (batch_size x n_tokens x word_i)
        """
        teacher_forcing_ratio = teacher_forcing_ratio[0]
        batch_size = image_vectors.shape[0]

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_out, hidden = self.encoder(image_vectors=image_vectors)

        logits = torch.zeros(batch_size, self.max_len - 1, self.n_tokens).to(
            self.device)  # to store logits (batch_size x max_len-1 x n_tokens) - '-1' - without <SOS>
        input_ = torch.ones((batch_size,), dtype=torch.int64).to(self.device) * SOS_IDX  # [batch_size]

        for t in range(1, self.max_len):

            # TODO pass state and input throw decoder
            logit, hidden, decoder_attention = self.decoder(input_=input_,
                                                            hidden=hidden,
                                                            encoder_outputs=enc_out)

            logits[:, t - 1, :] = logit  # (batch_size x max_len-1 x n_tokens)
            teacher_force = random.random() < teacher_forcing_ratio
            if topk == 1:
                top1 = logit.max(1)[1]  # -> [batch_size]
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
                topk_logits, topk_idxs = torch.topk(logits, k=topk, dim=2)  # -> (batch_size x max_len-1 x beam_szie)
                # top1 = topk_logits[:,:,torch.randint(0, topk, (1,))].squeeze(2) # -> (batch_size x max_len-1) - random sample version
                topk_logits = F.softmax(topk_logits, dim=2).cpu().numpy()  # -> (batch_size x max_len-1 x beam_szie)
                topk_idxs = topk_idxs.cpu()
                for i_bs in range(batch_size):
                    top1.append([np.random.choice(topk_idxs[i_bs, i_ml], 1, p=topk_logits[i_bs, i_ml]) for i_ml in
                                 range(max_len - 1)])  # weighted sample version
                top1 = torch.tensor(top1, dtype=torch.int64).to(self.device)
                probs = torch.gather(logits[:, :-1, :], dim=2, index=top1)
                top1 = top1.squeeze(2)
                probs = torch.pow(torch.prod(probs.squeeze(2), dim=1), 1 / (max_len - 1)).cpu().numpy()
        return top1, probs  # -> (batch_size x max_len-1) and (batch_size)

    def _init_weights(self):
        p = 0.08
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -p, p)

def fix_seed():
    seed = image_captioning['general']['SEED']
    torch.manual_seed(image_captioning['general']['SEED'])

fix_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = Encoder(device=DEVICE,
              src_len=image_captioning['model_parameters']['2_layers_attn']['SRC_LEN'],
              kernel_size=image_captioning['model_parameters']['2_layers_attn']['KERNEL_SIZE'],
              hid_dim=image_captioning['model_parameters']['2_layers_attn']['HIDDEN_SIZE'],
              num_layers=image_captioning['model_parameters']['2_layers_attn']['NUM_LAYERS'],
              cnn_feature_size=image_captioning['model_parameters']['2_layers_attn']['CNN_FEATURE_SIZE'],
              in_channels=image_captioning['model_parameters']['2_layers_attn']['IN_CHANNELS'],
              bidirectional=image_captioning['model_parameters']['2_layers_attn']['BIDIRECTIONAL'],
              dropout=image_captioning['model_parameters']['2_layers_attn']['DROPOUT'])

attention = Attention(hid_dim=image_captioning['model_parameters']['2_layers_attn']['HIDDEN_SIZE'],
                      num_layers=image_captioning['model_parameters']['2_layers_attn']['NUM_LAYERS'],
                      bidirectional=image_captioning['model_parameters']['2_layers_attn']['BIDIRECTIONAL'],
                      method=image_captioning['model_parameters']['2_layers_attn']['ATTENTION_METHOD'])

dec = LuongDecoderAttn(device=DEVICE,
                       n_tokens=VOCAB_SIZE,
                       emb_dim=image_captioning['model_parameters']['2_layers_attn']['EMB_DIM'],
                       hid_dim=image_captioning['model_parameters']['2_layers_attn']['HIDDEN_SIZE'],
                       attention=attention,
                       num_layers=image_captioning['model_parameters']['2_layers_attn']['NUM_LAYERS'],
                       bidirectional=image_captioning['model_parameters']['2_layers_attn']['BIDIRECTIONAL'],
                       dropout=image_captioning['model_parameters']['2_layers_attn']['DROPOUT'])

attn_2_layers = Seq2Seq(device=DEVICE,
                        encoder=enc,
                        decoder=dec,
                        n_tokens=VOCAB_SIZE,
                        max_len=image_captioning['general']['MAX_LEN']).to(DEVICE)

attn_2_layers = attn_2_layers.to(DEVICE)