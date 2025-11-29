import pandas as pd

drug_dict = pd.read_csv("../dataset/drugbank/drug_smiles.csv")

data_train = pd.read_csv("../dataset/drugbank/fold0/train.csv")

data_test = pd.read_csv("../dataset/drugbank/fold0/test.csv")

id2smiles = dict(zip(drug_dict['drug_id'], drug_dict['smiles']))

data_train['d1_smiles'] = data_train['d1'].map(id2smiles)
data_train['d2_smiles'] = data_train['d2'].map(id2smiles)

data_test['d1_smiles'] = data_test['d1'].map(id2smiles)
data_test['d2_smiles'] = data_test['d2'].map(id2smiles)

data_train.to_csv("../dataset/drugbank/train_f.csv", index=False)
data_test.to_csv("../dataset/drugbank/test_f.csv", index=False)


