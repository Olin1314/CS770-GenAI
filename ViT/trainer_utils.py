import os
import re
import torch
import logging


def save_checkpoint(model, epoch, outdir):
    """
    Saves the model checkpoint

    Args:
    model: The model used for the checkpoints
    epoch: Epoch number
    outdir : Base directory where the experiment is saved. Defaults to "checkpoints".
    
    Returns:
    None
    """
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch:3d}.pt')
    torch.save(model.state_dict(), cpfile)


def config_logger(path):
    """
    Sets the logger to log info in terminal and file.
    Args:
    path: Path to the log file.
    Returns:None
    """
    if os.path.exists(path) is True:
        os.remove(path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging into file
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging into console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def updated_checkpoint(model_dir="./checkpoints"):
    """
    Get the latest checkpoint file from a directory

    Args:
    model_dir: Path to the directory containing the checkpoints

    Returns:
    checkpoint: The latest checkpoint file
    """
    files = [f for f in os.listdir(model_dir) if 'model' in f]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    if len(files) == 0:
        return None
    return os.path.join(model_dir, files[-1])