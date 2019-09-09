import os
import torch
from tqdm import tqdm
from options import create_options
from data import create_dataloader, image_folder
from models import create_model

parser = argparse.ArgumentParser("Test")
parser.add_argument('--opt_name', required=True, help='options file name')
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--mode', default='test', help='test mode')
parser.add_argument('--data_dir', default='', help='input image dir')
args = parser.parse_args()

if __name__ == '__main__':
    # ----------------------------
    # get options [opt_name + '_options.py']
    options = create_options(args.opt_name)
    options.update_with_args(args)
    expr_dir = options.print_options()
    opt = options.parse()

    #-----------------------------
    # get dataloader [opt_dataset + '_dataset.py']
    if args.data_dir > 0:
        test_data_set = image_folder.ImageFolder(args.data_dir)
        test_data_loader = torch.utils.data.DataLoader(
            test_data_set,
            batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=int(opt['workers']))
    else:
        opt['mode'] = args.mode
        test_data_loader = create_dataloader(opt)
        test_dataset_size = len(test_data_loader)
        print("# testing images length", test_dataset_size)

    #-----------------------------
    # get model [opt_model_name + '_model.py']
    model = create_model(opt)
    info = model.setup(opt, expr_dir)
    global_step = info.get('step', 0)
    opt['start_epoch'] = info.get('epoch', 0)

    model.eval()
    with tqdm(total=len(test_data_loader)) as pbar:
        pbar.set_description(desc='Validate: ')
        for i, data in enumerate(test_data_loader):
            vis_ret = model.inference(data)
            pbar.update(1)
