import robomimic
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# make default BC config
config = config_factory(algo_name="bc")

# set config attributes here that you would like to update
config.experiment.name = "bc_horizon_subseq"
config.train.data = "../datasets/lift/ph/low_dim_v141.hdf5"
config.train.output_dir = "../bc_trained_subseq"
config.train.batch_size = 256
config.train.num_epochs = 500
config.algo.gmm.enabled = False

# Set the demonstration as subseqnece of the episode
config.train.seq_length = 20

# get torch device
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# launch training run
train(config, device=device)