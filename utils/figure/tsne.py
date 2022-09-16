import os
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.special import kl_div

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from egnn import EGNN_Network, predictor
from utils.utils import parse_args, set_seed


def TSNE_plot(visual='tsne'):
    args = parse_args()
    set_seed(args.seed)
    if args.data == 'lba':
        x_train, _, pos_train, y_train = torch.load(f'data/pdb/pdb_train_{args.split}.pt')
    else:
        x_train, pos_train, y_train = torch.load(f'data/pdb/lep_train.pt')
    train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                         dropout=args.dropout, global_linear_attn_every=1, norm_coors=True, coor_weights_clamp_value=2.,
                         aggregate=False).cuda()
    if 'PDB' in args.pretrain: model.out = predictor(args.dim).cuda()
    checkpoint = torch.load(args.save_path + args.pretrain)
    model.load_state_dict(checkpoint['model'])
    print(f'Model loading successfully from {args.pretrain}!')
    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

    feats, ys = [], []
    model.eval()
    for x, pos, y in train_loader:
        x, pos = x.long().cuda(), pos.float().cuda()
        mask = (x != 0)
        with torch.no_grad():
            out = model(x, pos, mask=mask)[1]
            if 'PDB' in args.pretrain:
                out = out[..., 0]
            else:
                if args.data == 'lep':
                    out = out.masked_fill(mask.unsqueeze(-1) == 0, 0)
                # 按理是要mask，但不mask效果更好
                out = torch.sum(out, dim=-2) / torch.sum(mask, dim=-1, keepdim=True)
            feats.append(out)
            ys.append(y)
    feats = torch.cat(feats, dim=0).cpu().numpy()
    ys = torch.cat(ys, dim=0).cpu().numpy()
    if args.data == 'lep':
        print('DB index:', davies_bouldin_score(feats, ys))
    # else:
    #     print('KL divergence:', kl_div(feats, ys.reshape(-1, 1)))
    if args.data == 'lep': feats = np.concatenate([feats[..., :1000], np.expand_dims(ys, axis=-1)], axis=-1)

    if visual == 'tsne':
        # TSNE可以调节iterations
        if '/' in args.pretrain: args.pretrain = args.pretrain.split('/')[-1]
        for iter in [1000, 5000, 10000]:
            tsne = TSNE(n_components=2, verbose=1, n_iter=iter)
            feats_ = tsne.fit_transform(feats)
            fig, ax = plt.subplots()
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.rcParams.update({'font.size': 14})
            if args.data == 'lba':
                plt.scatter(feats_[:, 0], feats_[:, 1], s=1.5, c=ys, alpha=0.8, cmap=cm.get_cmap('PuOr'))
                plt.colorbar(label='Affinity')
            else:
                scatter = ax.scatter(feats_[:, 0], feats_[:, 1], s=10, c=ys, alpha=0.8, cmap=cm.get_cmap('PuOr'))
                ax.add_artist(ax.legend(*scatter.legend_elements(), title='efficacy'))
            plt.savefig(f'{args.data}_{visual}_{iter}.pdf', bbox_inches='tight')
            print(f'Figure save at {visual}_{iter}.pdf!')
    elif visual == 'pca':
        pca = PCA(n_components=2, svd_solver='full')
        feats_ = pca.fit_transform(feats)
    elif visual == 'umap':
        # pip install umap-learn
        import umap
        feats_ = umap.UMAP().fit_transform(feats)

    if visual != 'tsne':
        fig, ax = plt.subplots()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.rcParams.update({'font.size': 14})
        if args.data == 'lba':
            plt.scatter(feats_[:, 0], feats_[:, 1], s=1.5, c=ys, alpha=0.8, cmap=cm.get_cmap('seismic'))
            plt.colorbar(label='Affinity')
        else:
            scatter = ax.scatter(feats_[:, 0], feats_[:, 1], s=10, c=ys, alpha=0.8, cmap=cm.get_cmap('seismic'))
            ax.add_artist(ax.legend(*scatter.legend_elements(), title='efficacy'))
        plt.savefig(f'{args.data}_{visual}.pdf', bbox_inches='tight')
        print(f'Figure save at {args.data}_{visual}.pdf!')


if __name__ == '__main__':
    TSNE_plot()
