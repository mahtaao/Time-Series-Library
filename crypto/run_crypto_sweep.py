#!/usr/bin/env python3
"""
Improved crypto forecasting script for W&B sweeps with proper model validation
"""

import os
import sys
import torch
import random
import numpy as np
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.crypto_exp_improved import CryptoExp_Long_Term_Forecast_Improved
from utils.print_args import print_args

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Crypto Forecasting with Sweep Support')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='crypto_sweep')
    parser.add_argument('--model', type=str, required=True, default='iTransformer')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom')
    parser.add_argument('--root_path', type=str, default='./crypto/dataset/')
    parser.add_argument('--data_path', type=str, default='crypto.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='label')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # model define
    parser.add_argument('--enc_in', type=int, default=24)
    parser.add_argument('--dec_in', type=int, default=24)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type2')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--inverse', action='store_true', default=True)
    parser.add_argument('--use_dtw', action='store_true', default=False)

    # W&B settings
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='crypto-forecasting', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='mahta-milaquebec', help='W&B entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--log_every_n_epochs', type=int, default=10, help='Log every N epochs')
    parser.add_argument('--log_predictions', action='store_true', default=True, help='Log sample predictions')
    parser.add_argument('--max_pred_samples', type=int, default=5, help='Maximum number of prediction samples to log')
    parser.add_argument('--run_data_integrity', action='store_true', default=True, help='Run data integrity checks and visualization')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    args = parser.parse_args()

    # Check GPU availability
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('GPU available: {}'.format(torch.cuda.get_device_name(args.gpu)))
        print('Using GPU: cuda:{}'.format(args.gpu))
    else:
        print('ERROR: No GPU available!')
        print('CUDA available:', torch.cuda.is_available())
        print('CUDA device count:', torch.cuda.device_count())
        print('Please ensure GPU is properly allocated in SLURM job')
        sys.exit(1)

    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    # Use the improved experiment class
    Exp = CryptoExp_Long_Term_Forecast_Improved

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            #     args.task_name,
            #     args.model_id,
            #     args.model,
            #     args.data,
            #     args.features,
            #     args.seq_len,
            #     args.label_len,
            #     args.pred_len,
            #     args.d_model,
            #     args.n_heads,
            #     args.e_layers,
            #     args.d_layers,
            #     args.d_ff,
            #     args.factor,
            #     args.embed,
            #     args.distil,
            #     args.des, ii)
            setting = '{}_{}'.format(args.model, args.des)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.task_name,
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)
        setting = '{}_{}'.format(args.model, args.des)


        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=0)
        torch.cuda.empty_cache() 