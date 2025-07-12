
from scipy.stats import spearmanr

import pandas as pd

# Replace 'your_file.csv' with your actual CSV file path
df = pd.read_csv('ip_scores_data.csv')




# Map Yes/No to 1/0
df['choice_numeric'] =  df['ipscore'] #df['choosed'].map({'Yes': 1, 'No': 0})
# df['Amp_norm'] = 
df['Kofta'] = df['Att_feats']
# df['Amp'] = df['Amp'] *100000
levels = df['level'].unique()
correlations = {}

for lvl in levels:


  

    subset = df[df['level'] == lvl]
    Kofta_unique = subset['Kofta'].nunique()
    choice_unique = subset['choice_numeric'].nunique()
    Kofta_std = subset['Kofta'].std()
    choice_std = subset['choice_numeric'].std()
    # print(subset['choice_numeric'])
    print(f"Level {lvl}: Koftaunique = {Kofta_unique}, choice unique = {choice_unique}")
    print(f"Level {lvl}: Koftastd = {Kofta_std:.4f}, choice std = {choice_std:.4f}")
    # print(subset['Kofta'])
    corr = subset['Kofta'].corr(subset['choice_numeric'])
    correlations[lvl] = corr

print("Correlation between choice and Koftaper level:")
for lvl, corr in correlations.items():
    print(f"Level {lvl}: correlation = {corr}")




import matplotlib.pyplot as plt
import seaborn as sns

# Suppose 'correlations' is the dict from before: {level: correlation_value}

# Convert to DataFrame for easy plotting
corr_df = pd.DataFrame({
    'level': list(correlations.keys()),
    'correlation': list(correlations.values())
})

# Plotting
plt.figure(figsize=(12, 6))
sns.boxplot(x='level', y='Kofta', hue='choosed', data=df)
plt.title('Ipscore Distribution by Choice Picked Across Levels')
plt.xlabel('Level')
plt.ylabel('Ipscore')
plt.legend(title='Choice Picked')
# plt.show()
# plt.show()
plt.savefig('correlation_by_level.png')




plt.figure(figsize=(8, 6))
sns.boxplot(x='ipscore', y='choice_numeric', data=df)
plt.title('Amp vs Choice (Boxplot)')
plt.xlabel('Choice (0 = No, 1 = Yes)')
plt.ylabel('Amp')
plt.grid(True)
plt.savefig('correlation_by_level.png')


# plt.figure(figsize=(12, 6))
# sns.stripplot(x='level', y='Kofta', hue='choosed', data=df, jitter=True, dodge=True)
# plt.title('Ipscore Scatter by Choice Picked Across Levels')
# plt.xlabel('Level')
# plt.ylabel('Ipscore')
# plt.legend(title='Choice Picked')
# plt.show()

# plt.savefig('scatter.png')




# from scipy.stats import spearmanr

# spearman_corrs = {}

# for lvl in levels:
#     subset = df[df['level'] == lvl]

#     if subset['choice_numeric'].nunique() > 1 and subset['Kofta'].nunique() > 1:
#         corr, pval = spearmanr(subset['Kofta'], subset['choice_numeric'])
#     else:
#         corr, pval = float('nan'), float('nan')

#     spearman_corrs[lvl] = (corr, pval)

# for lvl, (corr, pval) in spearman_corrs.items():
#     print(f"Level {lvl}: Spearman correlation = {corr:.3f}, p-value = {pval:.3g}")