import os
import time
import datetime
import shutil

import torch
import itertools
import collections
import pickle
import numpy as np
import subprocess

from models.build import build_model
from data.build import build_dataset, build_train_dataloader, build_val_dataloader
from solver.build import build_optimizer, build_lr_scheduler

from utils.logger import Logger
from utils.dist import init_distributed_model, is_main_process
from utils.seed import setup_seed
from utils.engine import train_one_epoch, evaluate


import custom_hessian
import torch.nn.functional as F
from torch.func import functional_call
def calc_full_hessian(model, criterion, data_iter, device):
    # model must be on device
    # data does not need to be on device
    num_param = sum(p.numel() for p in model.parameters())
    names = list(n for n, _ in model.named_parameters())
    
    def loss(*params):
        l = []
        for x, y in data_iter:
            y_hat = functional_call(model, {n: p for n, p in zip(names, params)}, x.to(device))
            l.append(
                criterion(y_hat, y.to(device))
            )
        return torch.mean(torch.stack(l, dim=0))

    H = torch.autograd.functional.hessian(loss, tuple(model.parameters()), create_graph=False)
    H = torch.cat([torch.cat([e.flatten() for e in Hpart]) for Hpart in H]) # flatten
    H = H.reshape(num_param, num_param)
    return H.cpu().numpy()


def main(args, commands_to_run_each_epoch=None):
    # init seed
    setup_seed(args)

    # init dist
    init_distributed_model(args)

    # init log
    logger = Logger(args)
    logger.log(args)

    # build dataset and dataloader
    train_data, val_data, n_classes = build_dataset(args)
    train_loader = build_train_dataloader(
        train_dataset=train_data,
        args=args
    )
    val_loader = build_val_dataloader(
        val_dataset=val_data,
        args=args
    )
    args.n_classes = n_classes
    logger.log(f'Train Data: {len(train_data)}, Test Data: {len(val_data)}.')

    # build model
    model = build_model(args)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    logger.log(f'Model: {args.model}')

    # build loss
    criterion = torch.nn.CrossEntropyLoss()

    # build solver
    if args.restart_path:
        # different from resume -- optimization and LR scheduler are reset
        checkpoint = torch.load(args.restart_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    optimizer, base_optimizer = build_optimizer(args, model=model_without_ddp)
    lr_scheduler = build_lr_scheduler(args, optimizer=base_optimizer)
    logger.log(f'Optimizer: {type(optimizer)}')
    logger.log(f'LR Scheduler: {type(lr_scheduler)}')

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # import ipdb; ipdb.set_trace()
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr_scheduler.step(args.start_epoch)
        logger.log(f'Resume training from {args.resume_path}.')
    
    # start train:
    logger.log(f'Start training for {args.epochs} Epochs.')
    start_training = time.time()
    max_acc = 0.0
    
    hessian_data = collections.defaultdict(list)

    do_hessian = args.hessian_store_full or args.hessian_calc_trace or args.hessian_top_n or args.hessian_calc_frobenius
    if do_hessian:
        _ds = train_data if args.hessian_n_samples < 0 else torch.utils.data.Subset(train_data, np.arange(args.hessian_n_samples))
        hessian_data_loader = build_val_dataloader(val_dataset=_ds, args=args)
        
    all_train_stats = []
    all_val_stats = []
    
    for epoch in range(args.start_epoch, args.epochs):
        start_epoch = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        run_id = None
        if args.save_dir:
            run_id = os.path.basename(args.save_dir)
            
        train_stats = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epoch=epoch, 
            logger=logger, log_freq=args.log_freq, use_closure=(args.opt[:4] == 'sam-' or args.opt[:4] == 'ssam' or args.opt[:4] == 'isam' or args.opt[:6] == 'frosam'),
            run_id=run_id
        )
        
        lr_scheduler.step(epoch)
        val_stats = evaluate(model, val_loader)

        at_epochs = [int(x) for x in args.hessian_at_epochs.split('|') if x]
        
        if do_hessian:
            assert at_epochs or (args.hessian_every_n_epochs > 0)
            
        if do_hessian and ((epoch in at_epochs) or (args.hessian_every_n_epochs > 0 and epoch % args.hessian_every_n_epochs == 0)):
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            hessian_data['epochs'].append(epoch)
            if args.hessian_store_full:
                h = calc_full_hessian(model, criterion, hessian_data_loader, 'cuda')
                hessian_data['full'].append(h)
            if args.hessian_calc_trace:
                o = custom_hessian.hessian(model, criterion, data=None, dataloader=hessian_data_loader, cuda=True)
                hessian_data['trace'].append(
                    o.trace()
                )
            if args.hessian_calc_frobenius:
                o = custom_hessian.hessian(model, criterion, data=None, dataloader=hessian_data_loader, cuda=True)
                hessian_data['frobenius'].append(
                    o.frobenius()
                )
            if args.hessian_top_n:
                o = custom_hessian.hessian(model, criterion, data=None, dataloader=hessian_data_loader, cuda=True)
                hessian_data['top_eigenvalues'].append(
                    o.eigenvalues(top_n=args.hessian_top_n)[0]
                )
            if args.hessian_calc_density:
                o = custom_hessian.hessian(model, criterion, data=None, dataloader=hessian_data_loader, cuda=True)
                hessian_data['density'].append(
                    o.density()
                )
                
        if epoch in args.save_checkpoints_at:
            if is_main_process:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(args.save_dir, f'checkpoint_{epoch}.pth'))

        
        if max_acc < val_stats["test_acc1"]:
            max_acc = val_stats["test_acc1"]
            if is_main_process:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, os.path.join(args.save_dir, 'checkpoint.pth'))
        
        logger.wandb_log(epoch=epoch, **train_stats)
        logger.wandb_log(epoch=epoch, **val_stats)

        all_train_stats.append(train_stats)
        all_val_stats.append(val_stats)

        output_data = {'train_stats': all_train_stats, 'val_stats': all_val_stats}
        if do_hessian:
            output_data['hessian_data'] = hessian_data
        
        if args.save_every_epoch:
            save_path = os.path.join(args.save_dir, f'latest.pickle')
            with open(save_path, 'wb') as f_:
                d = {'args': args}
                d.update(output_data)
                pickle.dump(d, f_)
        
        msg = ' '.join([
            'Epoch:{epoch}',
            'Train Loss:{train_loss:.4f}',
            'Train Acc1:{train_acc1:.4f}',
            'Train Acc5:{train_acc5:.4f}',
            'Test Loss:{test_loss:.4f}',
            'Test Acc1:{test_acc1:.4f}(Max:{max_acc:.4f})',
            'Test Acc5:{test_acc5:.4f}',
            'Time:{epoch_time:.3f}s'])
        logger.log(msg.format(epoch=epoch, **train_stats, **val_stats, max_acc=max_acc, epoch_time=time.time()-start_epoch))

        if commands_to_run_each_epoch:
            for c in commands_to_run_each_epoch:
                logger.log('Running ' + str(c))
                subprocess.run(c, check=True)
    
    logger.log('Train Finish. Max Test Acc1:{:.4f}'.format(max_acc))
    end_training = time.time()
    used_training = str(datetime.timedelta(seconds=end_training-start_training))
    logger.log('Training Time:{}'.format(used_training))
    logger.mv('{}_{:.4f}'.format(logger.logger_path, max_acc))
    return output_data



if __name__ == '__main__':
    from configs.defaulf_cfg import default_parser
    cfg_file = default_parser()
    args = cfg_file.get_args()
    torch.cuda.set_device(args.device)
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    outputs = {'args': args}
    outputs['results'] = main(args)
    with open(os.path.join(args.save_dir, 'all.pickle'), 'wb') as f_:
        pickle.dump(outputs, f_)





    