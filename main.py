import os
import copy
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from egnn import EGNN_Network, predictor
from utils.utils import parse_args, Logger, set_seed


def run_eval(args, model, loader, y_gt, ):
    model.eval()
    metric = 0
    y_pred = []
    with torch.no_grad():
        for x, pos, y in loader:
            x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
            mask = (x != 0)
            out = model(x, pos, mask=mask)[1][..., 0]
            if args.data == 'lep': out = torch.sigmoid(out)
            y_pred.append(out)
            if args.data == 'lba':
                metric += mse_loss(out, y, reduction='sum').item() / len(y_gt)
    y_pred = torch.cat(y_pred)
    if args.data == 'lba':
        spearman = stats.spearmanr(y_pred.cpu().numpy(), y_gt.numpy())[0]
        pearson = stats.pearsonr(y_pred.cpu().numpy(), y_gt.numpy())[0]
        return spearman, pearson, metric
    else:
        auroc = roc_auc_score(y_gt.numpy(), y_pred.cpu().numpy())
        auprc = average_precision_score(y_gt.numpy(), y_pred.cpu().numpy())
        return auroc, auprc, metric


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.data == 'lba':
        log = Logger(f'{args.save_path}pdbbind_{args.split}/', f'pdbind_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    else:
        log = Logger(f'{args.save_path}lep/', f'lep_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 1000
    # a large learning rate is helpful, the batch size of LBA is 16
    args.lr = 1e-4 * len(args.gpu.split(','))
    args.bs = 4 * len(args.gpu.split(','))

    if args.data == 'lba':
        x_train, _, pos_train, y_train = torch.load(f'data/pdb/pdb_train_{args.split}.pt')
        x_val, _, pos_val, y_val = torch.load(f'data/pdb/pdb_val_{args.split}.pt')
        if not args.unknown:
            x_test, _, pos_test, y_test = torch.load(f'data/pdb/pdb_test_{args.split}.pt')
        else:
            x_test, _, pos_test, y_test = torch.load(f'data/pdb/docking_test_{args.split}.pt')
    else:
        x_train, pos_train, y_train = torch.load(f'data/pdb/lep_train.pt')
        x_val, pos_val, y_val = torch.load(f'data/pdb/lep_val.pt')
        x_test, pos_test, y_test = torch.load(f'data/pdb/lep_test.pt')

    train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, pos_val, y_val), batch_size=args.bs * 2)
    test_loader = DataLoader(TensorDataset(x_test, pos_test, y_test), batch_size=args.bs * 2)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # a virtual atom is better than global pooling
    model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest, dropout=args.dropout, global_linear_attn_every=1,
                         norm_coors=True, coor_weights_clamp_value=2., aggregate=False).cuda()
    if args.pretrain:
        checkpoint = torch.load(args.save_path + args.pretrain)
        model.load_state_dict(checkpoint['model'])
        if args.linear_probe:
            for param in model.parameters():
                param.requires_grad = False
    else:
        args.pretrain = 'no_pre'
    model.aggregate = True
    model.out = predictor(args.dim).cuda()

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)
    if args.data == 'lba':
        criterion = torch.nn.MSELoss()
        best_metric = 1e9
    else:
        best_metric = 0
        criterion = torch.nn.BCELoss()
    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    if args.data == 'lba':
        lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)
    else:
        lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.6, patience=10, min_lr=5e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    log.logger.info(f'{"=" * 40} PDBbind {"=" * 40}\n'
                    f'Embed_dim: {args.dim}; Train: {len(x_train)}; Val: {len(x_val)}; Test: {len(x_test)}; Pre-train Model: {args.pretrain}'
                    f'\nData Split: {args.split}; Target: {args.data}; Batch_size: {args.bs}; Linear-probe: {args.linear_probe}\n{"=" * 40} Start Training {"=" * 40}')

    t0 = time()
    early_stop = 0
    try:
        for epoch in range(0, args.epochs):
            model.train()
            loss = 0.0
            t1 = time()
            for x, pos, y in train_loader:
                x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
                mask = (x != 0)
                out = model(x, pos, mask=mask)[1][..., 0]
                if args.data == 'lep': out = torch.sigmoid(out)

                loss_batch = criterion(out, y.float())
                loss += loss_batch.item() / (len(x_train) * args.bs)
                scaler.scale(loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if args.data == 'lba':
                spearman, pearson, metric = run_eval(args, model, val_loader, y_val)
                log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | RMSE: {:.3f} | Pearson: {:.3f} | Spearman: {:.3f} '
                                '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric ** 0.5, pearson, spearman, optimizer.param_groups[0]['lr'] * 1e5))
            else:
                auroc, auprc, _ = run_eval(args, model, val_loader, y_val)
                metric = auroc
                log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | AUROC: {:.3f} | AUPRC: {:.3f} '
                                '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, auroc, auprc, optimizer.param_groups[0]['lr'] * 1e5))
            lr_scheduler.step(metric)

            if (args.data == 'lba' and metric < best_metric) or (args.data == 'lep' and metric > best_metric):
                best_metric = metric
                best_model = copy.deepcopy(model)  # deep copy model
                best_epoch = epoch + 1
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 50: log.logger.info('Early Stopping!!! No Improvement on Loss for 50 Epochs.'); break
    except:
        log.logger.info('Training is interrupted.')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    if args.data == 'lba':
        spearman, pearson, metric = run_eval(args, best_model, test_loader, y_test)
    else:
        auroc, auprc, _ = run_eval(args, best_model, test_loader, y_test)
    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    if args.linear_probe: args.linear_probe = 'Linear'
    if args.data == 'lba':
        torch.save(checkpoint, args.save_path + f'PDB_{args.split}_{args.pretrain}_{args.linear_probe}.pt')
        log.logger.info(f'Save the best model as PDB_{args.split}_{args.pretrain}_{args.linear_probe}.pt.\nBest Epoch: {best_epoch} | '
                        f'RMSE: {metric ** 0.5} | Test Pearson: {spearman} | Test Spearman: {pearson}')
    else:
        torch.save(checkpoint, args.save_path + f'LEP_{args.split}_{args.pretrain}_{args.linear_probe}.pt')
        log.logger.info(f'Save the best model as LEP_{args.split}_{args.pretrain}_{args.linear_probe}.pt.\n'
                        f'Best Epoch: {best_epoch} | Test AUROC: {auroc} | Test AUPRC: {auprc}')


if __name__ == '__main__':
    main()
