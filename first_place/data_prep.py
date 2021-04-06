# =============================================================
# Load modules
# =============================================================

import os
import pandas as pd
from sklearn.model_selection import GroupKFold

# =============================================================
# Set parameters
# =============================================================

CURRENTDIR = os.path.abspath('')
PARENTDIR = os.path.dirname(CURRENTDIR)
RAW_DATA = os.path.join(PARENTDIR,'raw_data')
PROCESSED = 'processed'

# =============================================================
# Load data
# =============================================================

df_train = pd.read_csv(os.path.join(RAW_DATA,'train.csv'))
df_train_anno = pd.read_csv(os.path.join(RAW_DATA,'train_annotations.csv'))

# =============================================================
# Create folds
# =============================================================

target_cols = list(df_train.columns)[1:12]
Fold = GroupKFold(n_splits=5)
groups = df_train['PatientID'].values
for n, (train_index, val_index) in enumerate(Fold.split(df_train, df_train[target_cols], groups)):
    df_train.loc[val_index, 'fold'] = n
df_train['fold'] = df_train['fold'].astype(int)

# =============================================================
# Create annotations indicator variable
# =============================================================

anno_id = list(set(df_train_anno.StudyInstanceUID))
def id_indicator(id):
    if id in anno_id:
        return True
    else:
        return False
df_train['w_anno'] = df_train['StudyInstanceUID'].map(lambda x: id_indicator(x))

# ================================================================
# Save data to .csv
# ================================================================

print(f'\nShape of training dataframe: {df_train.shape}\n')
print('Observations per fold:\n\n',df_train.groupby('fold').size(),'\n')
df_train.to_csv(os.path.join(PROCESSED,'train_v2.csv'), index=False)
