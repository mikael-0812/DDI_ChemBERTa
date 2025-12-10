import pandas as pd

drug_dict = pd.read_csv("../dataset/drugbank/drug_smiles.csv")

data_train = pd.read_csv("../dataset/inductive_data/fold0/train.csv")

data_test = pd.read_csv("../dataset/inductive_data/fold0/s1.csv")

id2smiles = dict(zip(drug_dict['drug_id'], drug_dict['smiles']))

# data_train['d1_smiles'] = data_train['d1'].map(id2smiles)
# data_train['d2_smiles'] = data_train['d2'].map(id2smiles)
#
# data_test['d1_smiles'] = data_test['d1'].map(id2smiles)
# data_test['d2_smiles'] = data_test['d2'].map(id2smiles)

# data_train.to_csv("../dataset/inductive_data/train_f.csv", index=False)
# data_test.to_csv("../dataset/inductive_data/test_f.csv", index=False)

def get_flag(id):
    print(id)
    neg_id, flag = id.split('$')
    return neg_id, flag

neg_id, flag = data_train['split'].apply(get_flag)
flag = flag.lower()