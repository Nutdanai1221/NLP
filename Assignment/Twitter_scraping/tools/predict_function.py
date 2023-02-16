import pickle
from torchtext.vocab import Vocab
import spacy
from torch import nn
import torch
from torchtext.data.utils import get_tokenizer

# nlp = spacy.load("en_core_web_sm")
pad_ix = 1
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()
        #put padding_idx so asking the embedding layer to ignore padding
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emb_dim, 
                           hid_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, text, text_lengths):
        #text = [batch size, seq len]
        embedded = self.embedding(text)
        
        #++ pack sequence ++
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)
        
        #embedded = [batch size, seq len, embed dim]
        packed_output, (hn, cn) = self.lstm(packed_embedded)  #if no h0, all zeroes
        
        #++ unpack in case we need to use it ++
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        #output = [batch size, seq len, hidden dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
        #hn = [batch size, hidden dim * num directions]
        
        return self.fc(hn)
    
path = "model/LSTM.pt"

# load the Vocab object from file using pickle
with open('model/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    print(len(vocab))

input_dim  = len(vocab)
hid_dim    = 256
emb_dim    = 300       
output_dim = 2 
num_layers = 2
bidirectional = True
dropout = 0.5

model_load = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout)

model_load.load_state_dict(torch.load(path , map_location=torch.device('cpu')))
text_pipeline  = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1 #turn {1, 2, 3, 4} to {0, 1, 2, 3} for pytorch training 
mapping = vocab.get_itos()



def predict_text(test_str,model_load):

    text = torch.tensor(text_pipeline(test_str))
    text_list = [x.item() for x in text]
    [mapping[num] for num in text_list]
    text = text.reshape(1, -1)
    # text = torch.tensor(vocab_test_strdict).reshape(1, -1)
    text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)

    with torch.no_grad():
        output = model_load(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return predicted.item()
    
# print(predict_text("you are so good" ,model_load ))