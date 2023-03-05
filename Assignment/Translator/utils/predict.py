from torch import nn
import torch
import torchtext
from structure import Add_attention, Encoder, Decoder,Seq2SeqPackedAttention,Attention
# models architect

vocab_path ="model/vocab.pt"

# model.eval()
# with torch.no_grad():
#     output, attentions = model(src_text, text_length, trg_text, 0)

def Load_model(device, save_path = "models/last_Seq2SeqPackedAttention_additive.pt", params_path = "models/params.pt"):

    params = torch.load(params_path)

    attn = Attention(params['hid_dim'])
    enc  = Encoder(params['input_dim'], params['emb_dim'],  params['hid_dim'], params['dropout'])
    dec  = Decoder(params['output_dim'], params['emb_dim'],  params['hid_dim'], params['dropout'], attn)
    model = Seq2SeqPackedAttention(enc, dec, params['SRC_PAD_IDX'], device).to(device)

    model.load_state_dict(torch.load(save_path))

    return model

def translate(text):
    pass

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
    vocab = torch.load(vocab_path)
    model = Load_model(device)  
    model.eval()

    SRC_LANGUAGE = 'th'
    TRG_LANGUAGE = 'en'

    sec_text = "อยากไปนอนแล้วบาย"

    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0)