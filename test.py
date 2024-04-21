from LongSequenceGPT2 import LongSequenceGPT2, Config
from data_loader import load_data
import evaluate
from transformers import GPT2Tokenizer

df = load_data('cnn_dailymail', '1.0.0')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bleu = evaluate.load("bleu")
new_gpt = LongSequenceGPT2(Config)

# get generated string for each row
print("generated string for each row")
df['length'] = df['article'].apply(lambda x: new_gpt.get_tokenized_length(x))
df['generated_sentence'] = df.apply(lambda row: new_gpt.generate_text(row['input'], row['length']), axis=1)
df['generated_sentence_original'] = df.apply(lambda row: new_gpt.generate_text_original(row['input'], row['length'], 800), axis=1)

# calculate value for each row
print("calculate value for each row")

# score = bleu.compute(predictions=["how are you tony"], references=["how are you tony"])
# print(score)
df['bleu'] = df.apply(lambda row: bleu.compute(predictions=[row['generated_sentence']], references=[row['article']])['bleu'], axis=1)
df['bleu_original'] = df.apply(lambda row: bleu.compute(predictions=[row['generated_sentence_original']], references=[row['article']])['bleu'], axis=1)
print(df['bleu'])
print(df['bleu_original'])

df.to_csv('test.csv', index=False)