from torch import nn
import torch.nn.functional as F
from pythainlp.tokenize import word_tokenize

from datasets import load_dataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.manual_seed(9999)

tokenizer = GPT2Tokenizer.from_pretrained('Earth1221/GPT_Thai')

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("Earth1221/GPT_Thai", pad_token_id=tokenizer.eos_token_id)

def prediction_beam(text) :
    input_ids = tokenizer.encode(text, return_tensors='pt')
    beam_output = model.generate(
    input_ids,  
    max_length=50, 
    num_beams=3, 
    early_stopping=True
    )
    result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    # result = pipe(text, num_return_sequences=1)[0]["generated_text"]
    return result

def prediction_greedy(text) :
    input_ids = tokenizer.encode(text, return_tensors='pt')
    greedy_output = model.generate(input_ids, max_length=50)
    result = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    # result = pipe(text, num_return_sequences=1)[0]["generated_text"]
    return result