import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from atom3d.datasets import LMDBDataset


def pdb_loader(dict, mode='train', path='../../split-by-protein/data/', dist_bar=6):
    dataset = LMDBDataset(path + mode)
    x, pos, y = [], [], []
    for i in range(len(dataset)):
        struct = dataset[i]
        if struct['label'] == 'A':
            y += [1]
            key = 'atoms_active'
        else:
            y += [0]
            key = 'atoms_inactive'

        atoms = struct[key]
        lig = atoms[atoms.resname == 'UNK']
        protein = atoms[atoms.resname != 'UNK']
        dist = torch.cdist(tensor(protein[['x', 'y', 'z']].values), tensor(lig[['x', 'y', 'z']].values))
        mask = (torch.sum(dist <= dist_bar, dim=-1) > 0)
        mask = torch.cat((mask, torch.ones(len(lig)))).bool().numpy()
        atoms = atoms[mask]

        pos.append(tensor(atoms[['x', 'y', 'z']].values))
        x_tmp = []
        for m in atoms['element']:
            if m in dict.keys():
                x_tmp.append(dict[m])
            else:
                x_tmp.append(len(dict) + 1)
                dict[m] = len(dict) + 1
        x.append(tensor(x_tmp))

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    y = tensor(y)
    torch.save([x, pos, y], f'../data/pdb/lep_{mode}.pt')
    print(f'Loading {len(x)} data samples with {len(dict)} atom types.')
    return dict


if __name__ == '__main__':
    d = pdb_loader({}, mode='train')
    pdb_loader(d, mode='test')
