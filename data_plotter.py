import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv')

bleu_scores = df['bleu'].tolist()
bleu_scores_original = df['bleu_original'].tolist()

plt.hist(bleu_scores, bins=50, alpha=0.5, label='With summary', color='blue')
plt.hist(bleu_scores_original, bins=50, alpha=0.5, label='Without summary', color='red')

# Add legend
plt.legend()

# Add titles and labels
plt.title('Overlayed Histograms')
plt.xlabel('Value')
plt.ylabel('Frequency')

print(df['bleu'].mean())
print(df['bleu_original'].mean())

# Show plot
plt.show()

