import argparse
from typing import Dict, List, Any

import sys; sys.path.append('.')
from src.trainers.maskrcnn_trainer import MaskRCNNTrainer
from src.trainers.densepose_trainer import DensePoseRCNNTrainer
from firelab.config import Config


CONFIG_ARG_PREFIX = '--config.'


def run_trainer(args:argparse.Namespace, config_args:List[str]):
    # TODO: read some staff from command line and overwrite config
    config = Config.load('configs/densepose-rcnn.yml')

    if not args.local_rank is None:
        config.set('gpus', [args.local_rank])
    else:
        config.set('gpus', args.gpus)

    config.set('experiments_dir', args.experiments_dir)

    config_args = process_cli_config_args(config_args)
    config = config.overwrite(Config(config_args)) # Overwrite with CLI arguments

    trainer = DensePoseRCNNTrainer(config)

    if args.validate_only:
        print('Running validation only...')
        trainer.init()
        trainer.val_dataloader = trainer.train_dataloader
        trainer.validate()
    else:
        trainer.start()


def process_cli_config_args(config_args:List[str]) -> Dict:
    """Takes config args from the CLI and converts them to a dict"""
    # assert len(config_args) % 3 == 0, \
    #     "You should pass config args in [--config.arg_name arg_value arg_type] format"
    assert len(config_args) % 2 == 0, \
        "You should pass config args in [--config.arg_name arg_value] format"
    arg_names = [config_args[i] for i in range(0, len(config_args), 2)]
    arg_values = [config_args[i] for i in range(1, len(config_args), 2)]

    result = {}

    for name, value in zip(arg_names, arg_values):
        assert name.startswith(CONFIG_ARG_PREFIX), \
            f"Argument {name} is unkown and does not start with `config.` prefix. Cannot parse it."

        result[name[len(CONFIG_ARG_PREFIX):]] = infer_type_and_convert(value)

    return result


def infer_type_and_convert(value:str) -> Any:
    """
    Chances are high that this function should never exist...
    It tries to get a proper type and converts the value to it.
    """
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.isdigit():
        return int(value)
    elif is_float(value):
        return float(value)
    else:
        return value


def is_float(value:Any) -> bool:
    """One more dirty function: it checks if the string is float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpus', default=[0,1,2,3], type=int, nargs='+',
                        help='Which GPUs I should use (among those that are visible to me)')
    parser.add_argument('-d', '--experiments_dir', default='experiments', type=str,
                        help='Directory where to save checkpoints/logs/etc')
    parser.add_argument('--local_rank', type=int, help='Rank for distributed training')
    parser.add_argument('--validate_only', action='store_true', help='Flag denoting if we should just run validation')
    args, config_args = parser.parse_known_args()

    run_trainer(args, config_args)
