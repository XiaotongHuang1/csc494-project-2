from datasets import load_dataset
import pandas as pd
from transformers import GPT2Tokenizer, pipeline


def get_tokenized_length(text, tokenizer):
    # Get the length of the input sequence using tokenizer, will be a larger number
    # comparing to string split
    encoded_input = tokenizer(text, return_tensors='pt')
    sequence_length = encoded_input['input_ids'].shape[1]

    return sequence_length


def split_string_first(s):
    split_index = (len(s) + 1) // 2
    first_half = s[:split_index]
    return first_half


def split_string_second(s):
    split_index = (len(s) + 1) // 2
    second_half = s[split_index:]
    return second_half


def load_data(name, version):
    dataset = load_dataset(name, version)
    df = pd.DataFrame(data=dataset['test'])
    df['input'] = df['article'].apply(lambda x: split_string_first(x))
    df['target'] = df['article'].apply(lambda x: split_string_second(x))
    df['bleu'] = 0
    df['length'] = 0
    df = df.head(100)
    return df
