import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.utils.data
import os
import math
from time import time, strftime, localtime
from egnn import EGNN_Network, Classifier
from data.dataset import PDBMDDataset
from utils.utils import parse_args, Logger, set_seed


def train(args, model, optimizer, scaler, backprop=True, n_samples=16, max_samples=64, echo=False):
    if backprop:
        model.train()
    else:
        model.eval()

    loss_mse = nn.MSELoss()
    losses, n_frames = 0, 0,
    pdb_list = os.listdir('data/md/')[:max_samples]

    # n_sample does not significantly influence the speed of training
    for i in range(math.ceil(len(pdb_list) / n_samples)):
        if '.npy' not in ''.join(pdb_list[i * n_samples: (i + 1) * n_samples]): continue
        dataset = PDBMDDataset(backprop=backprop, pdb_list=pdb_list[i * n_samples: (i + 1) * n_samples],
                               data_dir=args.data_dir, max_len=args.max_len, noise=args.noise,
                               prompt=[int(x) for x in args.prompt.split(',')])
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=8)
        for data in loader:
            loc, charges, loc_end, prompt = [d.cuda() for d in data]
            mask = (charges != 0)
            with torch.cuda.amp.autocast(enabled=True):
                if backprop:
                    if args.ordering:
                        loc_pred, out_prev = model[0](charges, loc, mask=mask)
                        _, out_end = model[0](charges, loc_end, mask=mask)
                        order_pred, _ = model[1](out_prev, out_end)
                    else:
                        loc_pred, _ = model(charges, loc, mask=mask)
                else:
                    with torch.no_grad():
                        if args.ordering:
                            loc_pred, _ = model[0](charges, loc, mask=mask)
                        else:
                            loc_pred, _ = model(charges, loc, mask=mask)
                # loss may explode and be nan
                loss = loss_mse(loc_pred, loc_end)
            if args.ordering:
                order = torch.ones_like(order_pred).cuda()
                loss += F.binary_cross_entropy(order_pred, order)
            if backprop:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses += loss.item() * len(data[0])
            n_frames += len(loc)

        # clear cache，https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/3
        try:
            del loc, charges, loc_end, loc_pred, loss
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()
    if echo and backprop: print(f'Finishing with {n_frames} frames.')
    return losses, n_frames


def main():
    args = parse_args()
    set_seed(args.seed)
    folder_name = f'pretrain_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}'
    os.makedirs(args.save_path + folder_name)
    log = Logger(args.save_path + f'{folder_name}/', f'loss.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.epochs = 200
    args.bs = 16 * len(args.gpu.split(','))
    args.lr = 1e-4 / len(args.gpu.split(','))

    model = EGNN_Network(num_tokens=args.tokens, dim=args.dim, depth=args.depth, num_nearest_neighbors=args.num_nearest,
                         dropout=args.dropout, global_linear_attn_every=1, norm_coors=True, coor_weights_clamp_value=2.,
                         num_prompt=len(args.prompt.split(',')), aggregate=args.ordering).cuda()
    if args.ordering: classifier = Classifier(args.dim).cuda()
    if len(args.gpu) > 1:
        model = nn.DataParallel(model)
        if args.ordering:  classifier = torch.nn.DataParallel(classifier)
    if args.ordering: model = [model, classifier]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5, min_lr=args.min_lr)

    # amp: mixed precisions
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    log.logger.info(f'{"=" * 40} Pre-training MD {"=" * 40}\nOrdering: {args.ordering}; Noise: {args.noise}; Embed_dim: {args.dim}; '
                    f'Depth: {args.depth}; Max_len: {args.max_len};\nBatch_size: {args.bs}; GPU: {args.gpu}; Epochs: {args.epochs}')

    best_train_loss, best_test_loss, best_epoch = 1e15, 1e15, 0
    t0 = time()
    try:
        # training longer with a large batch size
        for epoch in range(0, args.epochs):
            t1 = time()
            train_loss, train_size = train(args, model, optimizer, scaler, True, max_samples=args.max_sample)
            val_loss, test_size = train(args, model, optimizer, scaler, False, max_samples=args.max_sample)
            if epoch == 0: log.logger.info(
                f'Train: {train_size}; Test: {test_size}\n{"=" * 40} Start Training {"=" * 40}')
            if val_loss < best_test_loss:
                best_test_loss = val_loss
                best_train_loss = train_loss
                best_epoch = epoch + 1
                best_model = model

                # 在训练中途而不是训练结束保存，以防训练中途报错
                checkpoint = {'model': best_model.state_dict(), 'epochs': args.epochs}
                if len(args.gpu) > 1: checkpoint['model'] = best_model.module.state_dict()
                if args.max_sample < 60:
                    torch.save(checkpoint, args.save_path + folder_name + f'/pre_n{args.max_sample}_e{best_epoch}.pt')
                else:
                    torch.save(checkpoint, args.save_path + folder_name + f'/pre_e{best_epoch}.pt')
                log.logger.info('Save the intermediate checkpoint! ')
            log.logger.info('Epoch: {} | Time: {:.2f}h | Loss: {:.3f} | MSE: {:.3f} | Lr: {:.3f}'.format(epoch + 1, (
                            time() - t1) / 3600, train_loss, val_loss, optimizer.param_groups[0]['lr'] * 1e5))
            lr_scheduler.step(val_loss)
    except:
        log.logger.exception("Exception Logged")
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(
        f'Save the best model as pre_e{best_epoch}.pt.\nBest Epoch: {best_epoch} | Train Loss: {best_train_loss:.3f} | '
        f'Val Loss: {best_test_loss:.3f}')


if __name__ == "__main__":
    main()
