import torch, torchdata, torchtext
from torch import nn
import torch.nn.functional as F
from pythainlp.tokenize import word_tokenize
import random, math, time
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

#make our work comparable if restarted the kernel
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads #make sure it's divisible....

        self.fc_q = nn.Linear(hid_dim,hid_dim) 
        self.fc_k = nn.Linear(hid_dim,hid_dim) 
        self.fc_v = nn.Linear(hid_dim,hid_dim) 

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, q, k, v, mask = None):
        batch_size = q.shape[0]
        
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)
        
        #Q, K, V = [b, l, h]
        #reshape them into head_dim
        #reshape them to [b, n_headm, l, head_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        #Q, K, V = [b, m_head, l, head_dim]

        #e = QK/sqrt(dk)
        e =  torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        #e = [b, n_heads, ql, kl]
        
        # torch.Size([64, 8, 50, 50])
        # torch.Size([64, 1, 1, 50, 256])

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)

        a = torch.softmax(e, dim=-1)
        #a = [batch size, n_heads, ql, kl]
                    
        #eV
        x = torch.matmul(self.dropout(a),V)
        #x : [b, n_heads, ql, head_di]

        x = x.permute(0, 2, 1, 3).contiguous()
        #x: [b, ql, n_heads, head_dim]

        #concat them together
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [b, ql, h]

        x = self.fc(x)
        #x = [b, ql, h]

        return x, a
    
class PositionwiseFeedforwardLayer(nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))    
     

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device, src_pad_idx,trg_pad_idx, max_length = 100):
        super().__init__()
        self.pos_emb = nn.Embedding(max_length, hid_dim)
        self.trg_emb = nn.Embedding(output_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
                            [
                            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                            for _ in range(n_layers)
                            ]
                            )
        self.fc = nn.Linear(hid_dim, output_dim)
        self.device = device
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_mask : [batch size, 1, 1, trg len]
        
        trg_len = trg_mask.shape[-1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device =self.device)).bool() #lower triangle
        #trg_sub_mask = [trg len, trg len]
        trg_mask = trg_mask & trg_sub_mask 
        #trg_mask : [batch size, 1, trg len, trg len]
        return trg_mask     
    def decode(self, trg, method='beam-search'):
        
        
        if method == 'beam-search':
            return self.beam_decode(trg)
        else:
            return self.greedy_decode(trg)     
    def greedy_decode(self, trg): # Get the best prop? maybe
        
        prediction= self.forward(trg)
        prediction = prediction.squeeze(0)
        prediction = prediction.argmax(1) 

        return prediction
    def forward(self, x):
        #src : = [batch size, trg len]
        #enc_src : hidden state from encoder = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = x.shape[0]
        trg_len = x.shape[1]
        
        src_mask = self.make_trg_mask(x)

        #pos
        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, trg len]

        pos_emb = self.pos_emb(pos) #[batch size, trg len, hid dim]
        trg_emb = self.trg_emb(x) #[batch size, trg len, hid dim]

        x = pos_emb + trg_emb * self.scale #[batch size, trg len, hid dim]
        x = self.dropout(x)
        
        for layer in self.layers: #output, hidden
            trg, attention = layer(x, src_mask)
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc(trg)
        #output = [batch size, trg len, output dim]

        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.norm_ff = nn.LayerNorm(hid_dim) #second yellow box
        self.norm_maskedatt = nn.LayerNorm(hid_dim) #first red box
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.ff = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, trg_mask):
        #trg      : [b, l, h]
        #enc_src  : [b, sl, h]
        #trg_mask : [b, 1, tl, tl]
        #src_mask : [b, 1, 1, sl]

        #1st box : mask multi, add & norm
        _trg, attention = self.self_attention(trg, trg, trg, trg_mask) #Q, K, V
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_maskedatt(_trg)

        #2rd box : ff, add & norm
        _trg    = self.ff(trg)
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_ff(_trg)

        return trg, attention
    
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    # hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src)

            # print(prediction.shape)
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
vocab_transform = torch.load("utils/transformer/vocab.pt")

output_dim  = len(vocab_transform)
hid_dim = 256
dec_layers = 12
dec_heads = 8
dec_pf_dim = 512
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

model = Decoder(output_dim, 
              hid_dim, 
              dec_layers, 
              dec_heads, 
              dec_pf_dim, 
              dec_dropout, 
              device,SRC_PAD_IDX,TRG_PAD_IDX).to(device)
model.apply(initialize_weights)

# prompt = 'ผมไปกินข้าว'
# max_seq_len = 30
# seed = 15648
# model.load_state_dict(torch.load('/content/drive/MyDrive/transformer/models_l/Decoder.pt', map_location=torch.device('cpu')))
# temperatures = [0.6789] 

# for temperature in temperatures:
#     generation = generate(prompt, max_seq_len, temperature, model, word_tokenize, 
#                           vocab_transform, device, seed)
#     print(str(temperature)+'\n'+''.join(generation)+'\n')
def prediction(text , model) :
    list_syn = []
    prompt = text
    max_seq_len = 30
    seed = 15648
    temperatures = [0.6789]
    model.load_state_dict(torch.load('utils/transformer/Decoder.pt', map_location=torch.device('cpu')))
    
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, word_tokenize, 
                            vocab_transform, device, seed)
        a = ''.join(generation)
        if a not in list_syn :
            list_syn.append(a)
    return list_syn


# print(prediction("วันนี้", model)) 