import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def convert_numeric(x):
    if pd.isna(x) or x == 'N/a': return np.nan
    try:
        return float(str(x).split('/')[0]) if '/' in str(x) else float(x)
    except: return np.nan


try:
    df = pd.read_csv("results.csv")
except:
    print("Error: results.csv file not found.")
    exit()


selected_columns = ['Gls', 'xG', 'Sh', 'Tkl', 'Int', 'Blocks']
columns = [c for c in selected_columns if c in df.columns]


for c in columns:
    df[c] = df[c].apply(convert_numeric)

df[columns] = df[columns].replace('N/a', np.nan)
df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')


with open("top_3.txt", "w", encoding="utf-8") as out:
    for col in columns:
        if df[col].dropna().empty: continue
        top = df[['Player', col]].dropna().sort_values(by=col, ascending=False).head(3)
        bottom = df[['Player', col]].dropna().sort_values(by=col).head(3)
        out.write(f"Statistic: {col}\nTop 3:\n")
        for _, r in top.iterrows(): out.write(f"  {r['Player']}: {r[col]}\n")
        out.write("Bottom 3:\n")
        for _, r in bottom.iterrows(): out.write(f"  {r['Player']}: {r[col]}\n")
        out.write("\n")


summaries = []
all_stats = {'Team': 'all'}
for col in columns:
    all_stats[f'Median of {col}'] = df[col].median()
    all_stats[f'Mean of {col}'] = df[col].mean()
    all_stats[f'Std of {col}'] = df[col].std()
summaries.append(all_stats)

for team in df['Squad'].unique():
    team_data = df[df['Squad'] == team]
    row = {'Team': team}
    for col in columns:
        row[f'Median of {col}'] = team_data[col].median()
        row[f'Mean of {col}'] = team_data[col].mean()
        row[f'Std of {col}'] = team_data[col].std()
    summaries.append(row)

pd.DataFrame(summaries).to_csv("results2.csv", index=False)


os.makedirs('histograms', exist_ok=True)
for col in columns:
    if df[col].dropna().empty: continue
    clean_name = re.sub(r'[\W]+', '_', col)
    plt.hist(df[col].dropna(), bins=20, edgecolor='black')
    plt.title(f'{col} - All')
    plt.savefig(f"histograms/{clean_name}_all.png")
    plt.close()
    for team in df['Squad'].unique():
        subset = df[df['Squad'] == team][col].dropna()
        if subset.empty: continue
        safe_team = re.sub(r'[^\w]', '_', team)[:50]
        plt.hist(subset, bins=20, edgecolor='black')
        plt.title(f'{col} - {team}')
        plt.savefig(f"histograms/{clean_name}_{safe_team}.png")
        plt.close()


best_by_metric = {}
for col in columns:
    if df[col].dropna().empty: continue
    max_idx = df[col].idxmax()
    if pd.isna(max_idx): continue
    max_team = df.loc[max_idx, 'Squad']
    max_val = df[col].max()
    best_by_metric[col] = {'Team': max_team, 'Value': max_val}

count_by_team = pd.Series([v['Team'] for v in best_by_metric.values()]).value_counts()
best_team = count_by_team.idxmax()
num_top_metrics = count_by_team.max()

with open('best_team_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Teams with the highest value per metric:\n")
    for metric, info in best_by_metric.items():
        f.write(f"{metric}: {info['Team']} ({info['Value']})\n")
    f.write(f"\nSummary:\n{best_team} is the best team with {num_top_metrics} leading statistics.\n")
