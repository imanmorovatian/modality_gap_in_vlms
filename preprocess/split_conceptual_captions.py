import numpy as np
import pandas as pd


df = pd.read_csv('data/annotations/conceptualCaptions/samples.csv', names=['id', 'caption'])
all_indices = df.index.tolist()
total = len(all_indices)

NO_VAL_SAMPLES = total // 10
NO_TEST_SAMPLES = total // 10
NO_TRAIN_SAMPLES = total - NO_VAL_SAMPLES - NO_TEST_SAMPLES

train_indices= np.random.choice(all_indices, size=NO_TRAIN_SAMPLES, replace=False)
df_train = df.loc[train_indices]

all_indices = [idx for idx in all_indices if idx not in train_indices]

val_indices = np.random.choice(all_indices, size=NO_VAL_SAMPLES, replace=False)
df_val = df.loc[val_indices]

test_indices = [idx for idx in all_indices if idx not in val_indices]
df_test = df.loc[test_indices]

df_train.to_csv('data/annotations/conceptualCaptions/train.csv', index=False)
df_val.to_csv('data/annotations/conceptualCaptions/val.csv', index=False)
df_test.to_csv('data/annotations/conceptualCaptions/test.csv', index=False)