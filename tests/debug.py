from data.mdataset import GraphDataset


toy_smiles = ['CCO', 'CCC']
ds = GraphDataset.from_smiles_list(toy_smiles, add_3d=False)
print(len(ds.graphs))