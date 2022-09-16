import torch
from dataloader_pdb import IUPAC_CODES


def distance(pos):
    dist = [torch.cdist(i[torch.nonzero(i[:, 0], as_tuple=True)[0]], i[torch.nonzero(i[:, 0], as_tuple=True)[0]]) for i
            in pos]
    dist = [torch.sum(i).item() / (2 * i.shape[0] * (i.shape[0] - 1)) for i in dist]
    return dist


def density():
    x_train, idx_train, pos_train, y_train = torch.load('../data/pdb_train_30.pt')
    x_test, idx_test, pos_test, y_test = torch.load('../data/pdb_test_30.pt')

    print('Min_len:', min(min([torch.count_nonzero(i) for i in x_train]).item(),
                          min([torch.count_nonzero(i) for i in x_test]).item()), 'Max_len:',
          max(x_train.shape[1], x_test.shape[1]))
    dist_train = distance(pos_train)
    dist_test = distance(pos_test)
    print('Density:', sum(dist_train + dist_test) / (len(dist_train) + len(dist_test)))


if __name__ == '__main__':
    print()







