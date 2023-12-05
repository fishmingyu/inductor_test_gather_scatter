import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Data provided
data = {
    'Dataset': ['Rand', 'Cora', 'Pubmed', 'Citeseer', 'PPI'],
    'Atomic': [0.274290, 0.015134, 0.125721, 0.013407, 0.044826],
    'CSR': [0.048228, 0.003448, 0.027408, 0.003432, 0.007810],
    'Segment': [0.060774, 0.003108, 0.027068, 0.002452, 0.008297]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Melting the DataFrame to long format suitable for seaborn
df_long = df.melt(id_vars=['Dataset'], var_name='Mode', value_name='Time')
# Let's use a different seaborn style and palette for a more aesthetically pleasing look
sns.set(style="whitegrid")
sns.set_palette("pastel")

# Drawing the figure with the new style
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(data=df_long, x='Dataset', y='Time', hue='Mode')
plt.title('Comparison of gather_scatter fusion on CPU')
plt.ylabel('Time (seconds)')
plt.xlabel('Dataset')
plt.legend(title='Mode')

plt.savefig("spmm")
