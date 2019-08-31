import argparse
import os
import random
import time
from contextlib import closing

import torch
from tqdm import tqdm
from data import create_dataloader
from models import create_model
from options import create_options
from util import data_utils, train_utils

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--opt_name', required=True, help='options file name')
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--gpu_ids',
                    type=str,
                    default='0,1,2,3',
                    help='gpu id used')
parser.add_argument('--continue',
                    action='store_true',
                    help='continue to train or train from start')
parser.add_argument('--mode', default='train', help='train mode')
args = parser.parse_args()

# set global variable
manual_seed = 99
print("Random Seed:", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

if __name__ == '__main__':
    # ----------------------------
    # get options [opt_name + '_options.py']
    options = create_options(args.opt_name)
    options.update_with_args(args)
    expr_dir = options.print_options()
    opt = options.parse()

    #-----------------------------
    # get dataloader [opt_dataset + '_dataset.py']
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print("# training images length ", dataset_size)
    opt['mode'] = 'test'
    test_data_loader = create_dataloader(opt)
    test_dataset_size = len(test_data_loader)
    print("# testing images length", test_dataset_size)
    opt['mode'] = 'train'

    #-----------------------------
    # get model [opt_model_name + '_model.py']
    model = create_model(opt)
    info = model.setup(opt, expr_dir)
    global_step = info.get('step', 0)
    opt['start_epoch'] = info.get('epoch', 0)

    # start training
    with closing(
            train_utils.MultiStepStatisticCollector(
                log_dir=expr_dir, global_step=global_step)) as stat_log:
        for epoch in range(opt['start_epoch'],
                           opt['niter'] + opt['niter_decay'] + 1):
            epoch_start_time = time.time()

            # train one pass
            with tqdm(total=len(data_loader)) as pbar:
                pbar.set_description(desc=f'epoch-{epoch}')
                for i, data in enumerate(data_loader):
                    iter_start_time = time.time()
                    model.set_input(data)

                    model.optimize_parameters()
                    if global_step % opt['print_freq'] == 0:
                        losses_ret = model.get_current_losses()
                        t_data = time.time() - iter_start_time
                        post_fix = ''
                        post_fix += f'time:{t_data:.3f}'
                        for k, v in losses_ret.items():
                            post_fix += f'{k} = {v:.3f} '
                        pbar.set_postfix_str(s=post_fix[:80])

                    if global_step % opt['log_freq'] == 0:
                        log_ret = model.get_current_log()
                        for a, d in log_ret.items():
                            for k, v in d.items():
                                stat_log.__getattr__('add_' + a)(k, v)

                    stat_log.next_step()
                    global_step = stat_log.count
                    pbar.update(1)

            checkpoint_path = os.path.join(expr_dir, "epoch" + str(epoch))
            model.save_nets(model.model_dict, {
                'epoch': epoch,
                'step': stat_log.count
            }, checkpoint_path)
            print(
                f'End of epoch {epoch} \t Time Taken: {time.time() - epoch_start_time} secs'
            )
            model.update_learning_rate(epoch)

            # validate one pass
            meters = []
            for i in range(len(model.loss_dict.keys())):
                meters.append(train_utils.AverageMeter())

            model.eval()
            with tqdm(total=len(test_data_loader)) as pbar:
                pbar.set_description(desc='Validate: ')
                for i, data in enumerate(test_data_loader):
                    model.set_input(data)
                    model.test()
                    losses = model.get_current_losses()
                    for index, key in enumerate(losses):
                        value = losses[key]
                        meters[index].update(value)
                    pbar.update(1)

                vis_ret = model.get_current_visuals()
                hist_ret = model.get_current_hist()
                for k, v in vis_ret.items():
                    stat_log.add_images('test_' + k, v)
                for k, v in hist_ret.items():
                    stat_log.add_histogram('test_' + k, v)

                post_fix = ''
                for i in range(len(meters)):
                    post_fix += f'{list(model.loss_dict.keys())[i]}={meters[i].avg:.3f}'
                    stat_log.add_scalar('test_' + list(model.loss_dict.keys())[i],
                                        meters[i].avg)
                post_fix += '**'
                pbar.set_postfix_str(s=post_fix)
                

    print('Training was successfully finished.')
