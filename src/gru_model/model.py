import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCaptionDecoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim):
        super(LightweightCaptionDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.decode_step = nn.GRUCell(embed_dim + encoder_dim, decoder_dim) 
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # 어텐션 모듈
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def attention(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

    def forward(self, encoder_out, captions):
        # encoder_out은 이미 [Batch, 49, 576] 형태로 들어온다고 가정
        embeddings = self.embedding(captions)
        h = self.init_h(encoder_out.mean(dim=1))
        
        seq_len = captions.size(1) - 1
        batch_size = captions.size(0)
        predictions = torch.zeros(batch_size, seq_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, seq_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(seq_len):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gru_input = torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1)
            h = self.decode_step(gru_input, h)
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha
        return predictions, alphas