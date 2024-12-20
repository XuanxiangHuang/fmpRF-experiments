import pandas as pd

ds_file = [
    'crx',
    'ecoli',
    'glass2',
    'hayes_roth',
    'house_votes_84',
    'iris',
    'mofn_3_7_10',
    'monk3',
    'new_thyroid',
    'agaricus_lepiota',
    'allbp',
    'ann_thyroid',
    'appendicitis',
    'collins',
    'hypothyroid',
    'ionosphere',
    'kr_vs_kp',
    'magic',
    'mushroom',
    'pendigits',
    'ring',
    'segmentation',
    'shuttle',
    'texture',
    'twonorm',
    'vowel',
    'waveform_21'
]

time_limit = 1200

for ds in ds_file:
    # Load the CSV file
    df = pd.read_csv(f'{ds}_results.csv')

    # Handle timeouts: set `answer` to 0 and count timeouts
    df['timeout'] = df['runtime'] > time_limit
    df.loc[df['timeout'], 'answer'] = 0

    # Filter out rows with timeouts
    df_no_timeout = df.query("timeout == False")

    # Compute statistics
    total_rows = len(df)
    num_answer_1 = (df['answer'] == 1).sum()
    percentage_answer_1 = (num_answer_1 / total_rows) * 100

    max_runtime = df_no_timeout['runtime'].max()
    mean_runtime = df_no_timeout['runtime'].mean()

    num_timeouts = df['timeout'].sum()

    # Print the formatted result
    print(f"{ds} & {percentage_answer_1:.0f}({num_timeouts}) & {max_runtime:.2f} & {mean_runtime:.2f} \\\\")
