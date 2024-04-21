from LongSequenceGPT2 import LongSequenceGPT2, Config

new_gpt = LongSequenceGPT2(Config)

text = """As aliens entered our planet"""
# new_gpt.generate_text(text, 2000)
new_gpt.generate_text_original(text, 2000)