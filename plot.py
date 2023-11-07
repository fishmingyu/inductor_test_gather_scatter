import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Data provided
data = {
    'Dataset': ['Rand', 'Cora', 'Pubmed', 'Citeseer', 'PPI'],
    'Eager Mode': [0.274290, 0.015134, 0.125721, 0.013407, 0.044826],
    'Graph Mode': [0.048228, 0.003448, 0.027408, 0.003432, 0.007810]
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
plt.title('Comparison of Atomic and CSR SpMM on CPU')
plt.ylabel('Time (seconds)')
plt.xlabel('Dataset')
plt.legend(title='Mode')

plt.savefig("spmm")
