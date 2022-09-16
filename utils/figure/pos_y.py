import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import scipy
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import davies_bouldin_score

from egnn import EGNN_Network
from utils.utils import parse_args, set_seed


def position_affinity_errorbar(step=None, fontsize=14, scatter=False):
    """ 检验预训练模型的position变化与affinity之间的关系 """
    args = parse_args()
    set_seed(args.seed)
    args.bs = 256 if args.data == 'lba' else 1024
    if step is None:
        step = 1 if args.data == 'lba' else 10  # LBA跑1步，LEP跑10步
    if os.path.exists(f'{args.data}_pos_y_{step}.pt'):
        pos_, ys = torch.load(f'{args.data}_pos_y_{step}.pt')
    else:
        if args.data == 'lba':
            x_train, _, pos_train, y_train = torch.load(f'data/pdb/pdb_train_{args.split}.pt')
        else:
            x_train, pos_train, y_train = torch.load(f'data/pdb/lep_train.pt')
        loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                             dropout=args.dropout, global_linear_attn_every=1, norm_coors=True, coor_weights_clamp_value=2.,
                             aggregate=False).cuda()
        checkpoint = torch.load(args.save_path + args.pretrain)
        model.load_state_dict(checkpoint['model'])
        print(f'Model loading successfully from {args.pretrain}! Running for {step} steps')
        if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

        pos_, ys = [], []
        model.eval()
        for x, pos, y in loader:
            x, pos = x.long().cuda(), pos.float().cuda()
            mask = (x != 0)
            with torch.no_grad():
                out = pos
                for i in range(step):
                    print(i)
                    out = model(x, out, mask=mask)[0]
                out = out.masked_fill(mask.unsqueeze(-1) == 0, 0)
                pos_.append(torch.sum((out - pos) ** 2, dim=-1).sum(-1) / mask.sum(-1))     # 求总和比求平均原子的移动效果要好
                ys.append(y)
        pos_ = torch.cat(pos_, dim=0).cpu().numpy() / step
        ys = torch.cat(ys, dim=0).cpu().numpy()
        torch.save([pos_, ys], f'{args.data}_pos_y_{step}.pt')

    if args.data == 'lba':
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pos_, ys)
        print(f'Slope: {slope}, intercept: {intercept}, r_value: {r_value}')
    else:
        db_index = davies_bouldin_score(pos_.reshape(-1, 1), ys)
        print('DB index:', db_index)

    if '/' in args.pretrain: args.pretrain = args.pretrain.split('/')[-1]
    plt.figure()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})
    if args.data == 'lba':
        if scatter:
            xy = np.vstack([pos_, ys])
            z = gaussian_kde(xy)(xy)
            plt.ylim([0, 14])
            plt.scatter(pos_, ys, c=z, s=50, alpha=0.8)
        else:
            bin_id, bins = pd.cut(pos_, bins=5, labels=[i for i in range(5)], retbins=True)
            df = pd.DataFrame({'ys': ys, 'bin': bin_id})
            bins = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
            plt.errorbar(bins, df.groupby('bin').mean()['ys'].values.tolist(), df.groupby('bin').std()['ys'].values.tolist(), linestyle='None', marker='^')
    else:
        df = pd.DataFrame({'pos': pos_, 'y': ys})
        for i in [0, 1]:
            subset = df[df['y'] == i]
            print(f'Class: {i}', 'Mean:', subset['pos'].mean(), 'Std:', subset['pos'].std())
            sns.kdeplot(subset['pos'], shade=True, linewidth=3, label=i)
        plt.legend(prop={'size': 14}, title='efficacy')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel(f'Predicted Change of Positions', fontsize=fontsize)
    if args.data == 'lba':
        plt.ylabel('Experimental Binding Affinities', fontsize=fontsize)
    else:
        plt.ylabel('Density', fontsize=fontsize)
    plt.savefig(f'{args.data}_pos_y_{step}.pdf', bbox_inches='tight')    # EPS格式不支持transparency
    print(f'Figure save at {args.data}_pos_y_{step}.pdf!')


if __name__ == '__main__':
    position_affinity_errorbar()
