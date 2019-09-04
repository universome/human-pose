import argparse
from typing import Dict

import sys; sys.path.append('.')
from src.trainers.maskrcnn_trainer import MaskRCNNTrainer
from src.trainers.densepose_trainer import DensePoseRCNNTrainer
from firelab.utils.fs_utils import load_config


def run_trainer(args:Dict):
    # TODO: read some staff from command line and overwrite config
    config = load_config('configs/densepose-rcnn.yml')

    if not args.get('local_rank') is None:
        config.set('gpus', [args['local_rank']])
    else:
        config.set('gpus', args['gpus'])
    config.set('experiments_dir', args['experiments_dir'])

    trainer = DensePoseRCNNTrainer(config)

    if args['validate_only']:
        print('Running validation only...')
        trainer.init()
        trainer.val_dataloader = trainer.train_dataloader
        trainer.validate()
    else:
        trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpus', default=[0,1,2,3], type=int, nargs='+',
                        help='Which GPUs I should use (among those that are visible to me)')
    parser.add_argument('-d', '--experiments_dir', default='experiments', type=str,
                        help='Directory where to save checkpoints/logs/etc')
    parser.add_argument('--local_rank', type=int, help='Rank for distributed training')
    parser.add_argument('--validate_only', action='store_true', help='Flag denoting if we should just run validation')
    args = vars(parser.parse_args())

    run_trainer(args)
