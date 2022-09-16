import os
import torch
from atom3d.datasets import LMDBDataset
from torch.nn.utils.rnn import pad_sequence


def pdb_selector(mode='train', split='30', atom3d_path='../../../../split-by-sequence-identity-*/data/', dock_path='../../../../PDBBind_processed'):
    atom3d_path = atom3d_path.replace('*', split)
    dataset = LMDBDataset(atom3d_path + mode)

    file_list = os.listdir(dock_path)
    zip_list = []
    for i in range(len(dataset)):
        id = dataset[i]['id']
        if id in file_list:
            zip_list.append(id)
    print(f'There are {len(zip_list)} items for {mode}.')

    import zipfile
    with zipfile.ZipFile(f'../../../../dock_{mode}.zip', 'w') as zipMe:
        for file in zip_list:
            path = f'{dock_path}/{file}'
            for root, dirs, files in os.walk(path):
                for f in files:
                    zipMe.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), os.path.join(path, '..')))


def dock_structure(mode='test', atom3d_path='data/split-by-sequence-identity-30/data/', dock_path='data/dock_'):
    atom_dict = {'N': 1, 'H': 2, 'C': 3, 'O': 4, 'P': 5, 'S': 6, 'MG': 7, 'NA': 8, 'ZN': 9, 'F': 10, 'Cl': 11, 'Br': 12, 'Ca': 13, 'Fe': 14, 'I': 15, 'Mn': 16, 'Co': 17, 'Cu': 18,
                 'Sr': 19, 'K': 20, 'Cs': 21, 'Ni': 22, 'Cd': 23}  # 和dataloder_pdb的dict一致

    dataset = LMDBDataset(atom3d_path + mode)
    file_list = os.listdir(dock_path + mode)

    x, pos, y = [], [], []
    for i in range(len(dataset)):
        id = dataset[i]['id']
        if id not in file_list: continue

        l_pos, l_x = [], []
        with open(f'data/results/{id}/lig_equibind_corrected.sdf') as file:
            for line in file:
                if len(line) < 45 or not line.startswith(' '): continue
                line = line.split()
                l_pos.append([float(i) for i in line[:3]])
                l_x.append(atom_dict[line[3]])
        l_pos = torch.tensor(l_pos)

        r_pos, r_x = [], []
        with open(f'{dock_path}{mode}/{id}/{id}_protein_processed.pdb') as file:
            for line in file:
                if len(line) < 50 or line.startswith('R'): continue

                line = line.split()
                tmp = torch.tensor([float(i) for i in line[6: 9]])
                dist = torch.cdist(tmp.unsqueeze(0), l_pos)
                if torch.sum(dist < 6, dim=-1) > 0:
                    r_pos.append(tmp)
                    r_x.append(atom_dict[line[-1]])
        if len(r_pos) > 0:
            pos.append(torch.cat([l_pos, torch.stack(r_pos, dim=0)]))
        else:
            continue    # 掠过没有pocket的样本
        x.append(torch.cat([torch.LongTensor(l_x), torch.LongTensor(r_x)], dim=0))
        y.append(dataset[i]['scores']['neglog_aff'])

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    y = torch.tensor(y)
    torch.save([x, [], pos, y], f'docking_{mode}_30.pt')
    print(f'Loading {len(x)} data samples.')


if __name__ == '__main__':
    dock_structure()

