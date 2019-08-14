import sys; sys.path.append('.')
from src.trainers.maskrcnn_trainer import MaskRCNNTrainer
from src.trainers.densepose_trainer import DensePoseRCNNTrainer
from firelab.utils.fs_utils import load_config


def main():
    config = load_config('configs/densepose-rcnn.yml')

    # TODO: read this from command line, because I am not the only one in the project
    config.set('available_gpus', [8])
    config.set('experiments_dir', 'densepose-experiments')

    trainer = DensePoseRCNNTrainer(config)
    trainer.start()


if __name__ == '__main__':
    main()
