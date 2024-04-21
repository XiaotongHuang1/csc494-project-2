from dataclasses import dataclass
import torch
from transformers import GPT2Tokenizer, pipeline
import json


@dataclass
class Config:
    summerizer_name: str = 'facebook/bart-large-cnn'
    generator_name: str = 'gpt2-large'
    tokenizer_name: str = 'gpt2-large'
    max_sequence_length: int = 1024
    max_summarizer_ratio: float = 0.5
    min_summary_length: int = 100
    max_summary_length: int = 512
    summary_threshold: int = 800  # If sequence greater than this length, use summarizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _josn_export(dictionary, filename):
    # export the dictionary to json file
    filepath = "./data/" + filename
    with open(filepath, 'w') as f:
        json.dump(dictionary, f)


class LongSequenceGPT2:
    def __init__(self, config):

        self.max_summarizer_ratio = config.max_summarizer_ratio
        self.max_sequence_length = config.max_sequence_length
        self.min_summary_length = config.min_summary_length
        self.max_summary_length = config.max_summary_length
        self.summary_threshold = config.summary_threshold

        self.tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_name)

        self.summarizer = pipeline("summarization",
                                   model=config.summerizer_name,
                                   max_length=self.max_summary_length,
                                   min_length=self.min_summary_length,
                                   device=config.device)

        # only get
        self.generator = pipeline('text-generation',
                                  model=config.generator_name,
                                  max_length=config.max_sequence_length,
                                  return_full_text=False,
                                  device=config.device)

    def _get_words(self, text):
        # Get the length of the input sequence using tokenizer, will be a larger number
        # comparing to string split
        encoded_input = self.tokenizer(text, return_tensors='pt')
        sequence_length = encoded_input['input_ids'].shape[1]
        words = text.split()
        words_length = len(words)

        return encoded_input, sequence_length, words, words_length

    def get_tokenized_length(self, text):
        # Get the length of the input sequence using tokenizer, will be a larger number
        # comparing to string split
        encoded_input = self.tokenizer(text, return_tensors='pt')
        sequence_length = encoded_input['input_ids'].shape[1]

        return sequence_length

    def partition_text(self, text, N):
        encoded_input, sequence_length, words, words_length = self._get_words(text)
        if sequence_length <= N:
            return text
        else:
            reversed_seg_point = int(N * 0.8)
            segment2 = words[-reversed_seg_point:]
            segment2_str = ' '.join(segment2)
            return segment2_str

    def summarize_text(self, text):
        encoded_input, sequence_length, words, words_length = self._get_words(text)
        summary_list = []

        # segment the input sequence if the sequence length exceed the max limit
        while sequence_length > self.summary_threshold:
            segment_point = self.summary_threshold if words_length > self.summary_threshold else words_length
            segment_point = int(segment_point * self.max_summarizer_ratio)
            segment1 = words[:segment_point]
            segment2 = words[segment_point:]
            segment1_str = ' '.join(segment1)
            segment2_str = ' '.join(segment2)

            # summarize the first segment while keep the second segment
            segment1_summary = self.summarizer(segment1_str,
                                               max_length=self.max_summary_length,
                                               min_length=self.min_summary_length)

            segment1_summary_str = segment1_summary[0]['summary_text']
            new_text = segment1_summary_str + " <EOSS> " + segment2_str
            dictionary = {
                'segment1': segment1_str,
                'segment2': segment2_str,
                'segment1_summary': segment1_summary_str,
                'segment2_summary': segment2_str,
                'new_text': new_text
            }
            summary_list.append(dictionary)
            encoded_input, sequence_length, words, words_length = self._get_words(new_text)

        # _josn_export(dictionary, f'summary({text[:10]}...{text[-5:]}).json')
        final_str = ' '.join(words)
        return final_str

    def generate_text_original(self, text, length, N):
        original_text = text
        sequence_length = self.get_tokenized_length(text)
        generated_list = []

        while sequence_length < length:
            attention_text = self.partition_text(text, N)
            generated = self.generator(attention_text, truncation=True)[0]['generated_text']
            generated_list.append(generated)
            text = text + ' ' + generated
            sequence_length += self.get_tokenized_length(generated)

        dictionary = {
            'original_text': original_text,
            'final_text': text,
            'generate_list': generated_list
        }

        _josn_export(dictionary, f'original_generation({text[:10]}...).json')
        return text

    def generate_text(self, text, length):
        original_text = text
        encoded_input, sequence_length, words, words_length = self._get_words(text)
        total_sequence_length = sequence_length
        generated_list = []

        # keep generating until reaches the length
        while total_sequence_length < length:

            # if current sequence is greater than the summary threadshold, do summarization
            if sequence_length > self.summary_threshold:
                text = self.summarize_text(text)

            generated = self.generator(text)[0]['generated_text']
            generated_list.append(generated)
            text = text + " " + generated
            encoded_input, sequence_length, words, words_length = self._get_words(text)
            total_sequence_length += self.get_tokenized_length(generated)

        final_text = original_text + ' '.join(generated_list)
        dictionary = {
            'original_text': original_text,
            'final_text': final_text,
            'generate_list': generated_list
        }

        _josn_export(dictionary, f'generation({text[:10]}...).json')
        return final_text
