from firelab.manager import run
from firelab.config import Config
import sys; sys.path.append('.')
from src.trainers.maskrcnn_trainer import MaskRCNNTrainer
from firelab.utils.fs_utils import load_config

args = Config({
    'config_path': 'configs/densepose-rcnn.yml',
    'stay_after_training': True,
    'tb_port': 13001
})
# args.tb_port = None

run('start', args)

# config = load_config('configs/mask-rcnn.yml')
# config.set('firelab', {
#     'device_name': 'cuda:8',
#     'exp_name': 'debug-mask-rcnn',
#     'available_gpus': [8],
#     'checkpoints_path': 'debug/checkpoints',
#     'logs_path': 'debug/logs',
#     'summary_path': 'debug/summary.yml'
# })
#
# trainer = MaskRCNNTrainer(config)
# trainer.start()
