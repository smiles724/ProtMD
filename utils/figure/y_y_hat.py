import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from egnn import EGNN_Network
from utils.utils import parse_args, set_seed

line = False
fontsize = 14
args = parse_args()
set_seed(args.seed)

if args.data == 'lba':
    args.pretrain = 'PDB_30_no_pre_False.pt'
    x_train, _, pos_train, y_train = torch.load(f'data/pdb/pdb_train_{args.split}.pt')
else:
    args.pretrain = 'LEP_30_no_pre_False.pt'
    x_train, pos_train, y_train = torch.load(f'data/pdb/lep_train.pt')
train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs * 2, shuffle=False)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                     dropout=args.dropout, global_linear_attn_every=1, norm_coors=True, coor_weights_clamp_value=2.,
                     aggregate=True).cuda()
checkpoint = torch.load(args.save_path + args.pretrain)
model.load_state_dict(checkpoint['model'])
print(f'Model loading successfully from {args.pretrain}!')
if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

pred = []
model.eval()
for x, pos, y in train_loader:
    x, pos = x.long().cuda(), pos.float().cuda()
    mask = (x != 0)
    with torch.no_grad():
        out = model(x, pos, mask=mask)[1][..., 0]
        if args.data == 'lep': out = torch.sigmoid(out)
        pred.append(out)
pred = torch.cat(pred, dim=0).cpu().numpy()
ys = y_train.numpy()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.rcParams.update({'font.size': fontsize})

if args.data == 'lba':
    xy = np.vstack([pred, ys])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(pred, ys, c=z, s=20, alpha=0.8)
    if line:
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, linestyle='--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')

    ax.set_xlabel('Predicted Affinities', fontsize=fontsize)
    ax.set_ylabel('Experimental Affinities', fontsize=fontsize)
    ax.set_xlim([1, 11])
    ax.set_ylim([1, 13])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
else:
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(ys, pred)
    roc_auc = metrics.roc_auc_score(ys, pred)
    plt.plot(fpr, tpr, 'red', label='ROC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

plt.savefig(f'y_y_hat.png', bbox_inches='tight')
















