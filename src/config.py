import argparse
import copy
import os
import numpy as np
import random

# Ensure deterministic CuBLAS behavior when torch.use_deterministic_algorithms is enabled
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

from save_and_load import create_save_dir
from data_setup import default_root

BASE_DEFAULT_ARGS = {
    "dataset": "celeba-Blond_Hair-Male",
    "datapath": default_root() + os.sep,
    "corruptedcifarunbiased_root": os.path.join(default_root(), "corrupted_cifar_unbiased"),
    "waterbirds_root": os.path.join(default_root(), "waterbirds_unbiased"),
    "seed": None,   # no default: a seed must always be chosen explicitly
    "device": "cuda:0",
    "projectName": None,
    "wandb_log": True,
    "log_ph_training": True,
    "use_tqdm": True,
    "train_private": False,
    "alpha": 1,
    "nb_epochs": 100,
    "warmup_epochs_model": 5,
    "refresh_epochs_model": 5,
    "batch_size": 100,
    "lr": 0.01,
    "mom_sgd": 0.9,
    "wd": 0.0001,
    "patience": 5,
    "cooldown": 5,
    "sched_factor": 0.1,
    "threshold_sched": 0.001,
    "min_lr": 0.0001,
    "head_types": "linear",
    "epochs_wo_ph": 5,
    "minimize_MI": True,
    "used_phs": "all",
    "nb_epochs_ph": "100,100,100,100",
    "refresh_epochs_ph": "10,10,10,10",
    "batch_size_ph": "100,100,100,100",
    "lr_ph": "0.01,0.01,0.01,0.01",
    "mom_sgd_ph": "0.9,0.9,0.9,0.9",
    "wd_ph": "0.0001,0.0001,0.0001,0.0001",
    "patience_ph": "5,5,5,5",
    "threshold_sched_ph": "0.001,0.001,0.001,0.001",
    "cooldown_ph": "0,0,0,0",
    "sched_factor_ph": "0.1,0.1,0.1,0.1",
    "min_lr_ph": "0.0001,0.0001,0.0001,0.0001",
    "gamma": None,
    "train_ph_each_epoch": True,
    "max_ph_deterioration": 1.5,
    "train_PHs_every_sparsity": False,
    "pruning_retraining": "none",
    "model": "resnet18",
    "pretrained": False,
    "reload_config": None,
    "save_model": True,
    "save_path": "models/",
    "save_each_epoch": False,
    "save_only_best_models": False,
    "prune": False,
    "blocks_to_prune": "0,1,2,3",
    "prune_block_by_block": True,
    "pruning_method": "local_structured",
    "pruning_amount": "0.5,0.5,0.5,0.5",
    "final_sparsity": "0.999,0.999,0.999,0.999",
    "criterion_best": "val_loss_and_private",
    "max_deter_pruning": 1.5,
    "max_loop_without_imp": 3,
    "pruning_criterion": "weight",
    "Nsparsities": 5000,
    "ph_training_fq": 5,
}


def str_to_list(s, dtype=int):
    """Convert a comma-separated string to a list of the specified type."""
    if isinstance(s, list):
        return s
    if isinstance(s, str) and "," in s:
        return [dtype(x.strip()) for x in s.split(",")]
    return s

def handle_all(args, nb_phs=None):
    """
    Ensure that all parameters have the same length as used_phs.
    If a parameter has fewer values than used_phs, fill the missing values with:
    1. The first item of the parameter if it's a list
    2. The parameter value itself if it's a single value
    
    If args.used_phs is "all" and nb_phs is provided, set args.used_phs to list(range(nb_phs)).
    """
    if isinstance(args.used_phs, str) and args.used_phs == "all" and nb_phs is not None:
        args.used_phs = list(range(nb_phs))
    
    params_to_check = [
        'nb_epochs_ph', 'refresh_epochs_ph', 'batch_size_ph', 'lr_ph', 
        'mom_sgd_ph', 'wd_ph', 'patience_ph', 'threshold_sched_ph', 
        'cooldown_ph', 'sched_factor_ph', 'min_lr_ph', 'gamma'
    ]
    
    used_phs_len = len(args.used_phs)
    
    # Ensure each parameter has the same length as used_phs
    for param_name in params_to_check:
        param_value = getattr(args, param_name, None)
        
        if param_value is None:
            continue
        
        if isinstance(param_value, list) and len(param_value) < used_phs_len:
            fill_value = param_value[0] if param_value else None
            
            param_value.extend([fill_value] * (used_phs_len - len(param_value)))
            
            setattr(args, param_name, param_value)

def parse_args(from_function=False, kwargs=None):
    """
    Parse command line arguments or create a default argument namespace.
    If `from_function` is True, it creates a default namespace with predefined values.
    If `from_function` is False, it uses argparse to parse command line arguments.
    If `kwargs` is provided, it updates the default values with the provided keyword arguments.
    Command-line parameters always take priority over defaults and ``kwargs``.
    """

    kwargs = kwargs or {}
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs should be a dictionary")

    default_args = copy.deepcopy(BASE_DEFAULT_ARGS)
    for key, value in kwargs.items():
        if key not in default_args:
            raise ValueError(f"Unknown argument: {key}")
        default_args[key] = value

    if from_function:
        args = argparse.Namespace(**default_args)
    else:
        parser = argparse.ArgumentParser(description="Train model with privacy heads")
        parser.add_argument("--dataset",              type=str,   default=default_args["dataset"],             help="Dataset to use")
        parser.add_argument("--datapath",             type=str,   default=default_args["datapath"],            help="Path to data directory")
        parser.add_argument(
            "--corruptedcifarunbiased_root",
            type=str,
            default=default_args["corruptedcifarunbiased_root"],
            help="Base path for the unbiased corrupted CIFAR-10-C dataset (train/val/test)",
        )
        parser.add_argument(
            "--waterbirds_root",
            type=str,
            default=default_args["waterbirds_root"],
            help="Base path for the unbiased Waterbirds dataset (train/val/test)",
        )
        parser.add_argument("--seed",                 type=int,   required=(default_args["seed"] is None),
                            default=default_args["seed"], help="Random seed; no default, choose and report it explicitly")
        parser.add_argument("--device",               type=str,   default=default_args["device"],             help="Device to use")
        parser.add_argument("--projectName",          type=str, required=True, help="Weights & Biases project name")
        parser.add_argument("--wandb_log",            action='store_true', default=default_args["wandb_log"], help="Log to wandb")
        parser.add_argument("--log_ph_training",      action='store_true', default=default_args["log_ph_training"], help="Log ph training")
        parser.add_argument('--use_tqdm',             action='store_true',                                      help="Use tqdm for progress bar")
        parser.add_argument('--no-tqdm',              action='store_false', dest='use_tqdm',                    help="Don't use tqdm for progress bar")

        # Classifier parameters
        parser.add_argument("--train_private",        action='store_true', default= default_args["train_private"],  help="Decide to train the classifier on the private label")
        parser.add_argument("--alpha",                type=float, default=default_args["alpha"],              help="Weight of the task loss in the total loss")
        parser.add_argument("--nb_epochs",            type=int,   default=default_args["nb_epochs"],          help="Number of epochs for main model")
        parser.add_argument("--warmup_epochs_model",  type=int,   default=default_args["warmup_epochs_model"],help="Number of warmup epochs for main model")
        parser.add_argument("--refresh_epochs_model", type=int,   default=default_args["refresh_epochs_model"],help="Batch size for main model")
        parser.add_argument("--batch_size",           type=int,   default=default_args["batch_size"],         help="Batch size for main model")
        parser.add_argument("--lr",                   type=float, default=default_args["lr"],                 help="Learning rate for main model")
        parser.add_argument("--mom_sgd",              type=float, default=default_args["mom_sgd"],            help="SGD momentum for the main model")
        parser.add_argument("--wd",                   type=float, default=default_args["wd"],                 help="Weight decay for the main model")
        parser.add_argument("--patience",             type=int,   default=default_args["patience"],           help="Patience for the main model learning rate scheduler")
        parser.add_argument("--cooldown",             type=int,   default=default_args["cooldown"],           help="Cooldown for the main model learning rate scheduler")
        parser.add_argument("--sched_factor",         type=float, default=default_args["sched_factor"],       help="Factor for the main model learning rate scheduler")
        parser.add_argument("--threshold_sched",      type=float, default=default_args["threshold_sched"],    help="Threshold for the main model learning rate scheduler")
        parser.add_argument("--min_lr",               type=float, default=default_args["min_lr"],             help="Minimum learning rate for the main model")

        # Privacy head parameters
        parser.add_argument("--head_types",           type=str,   default=default_args["head_types"],         help="Type of privacy head", choices=["linear", "double_linear"])
        parser.add_argument("--minimize_MI",          action='store_true', default= default_args["minimize_MI"], help="Minimize the MI between the private label and the phs outputs")
        parser.add_argument("--used_phs",             type=str,   default=default_args["used_phs"],           help="List of ph indices to use (comma-separated, e.g. --used_phs 0,1,2,3)")
        parser.add_argument("--nb_epochs_ph",         type=str,   default=default_args["nb_epochs_ph"],       help="Number of epochs for privacy heads (comma-separated)")
        parser.add_argument("--ph_training_fq",       type=int,   default=default_args["ph_training_fq"],     help="Number of epochs between each ph training")
        parser.add_argument("--refresh_epochs_ph",    type=str,   default=default_args["refresh_epochs_ph"],  help="Number of refresh epochs for privacy heads (comma-separated)")
        parser.add_argument("--batch_size_ph",        type=str,   default=default_args["batch_size_ph"],      help="Batch size for privacy heads (comma-separated)")
        parser.add_argument("--lr_ph",                type=str,   default=default_args["lr_ph"],              help="Learning rate for privacy heads (comma-separated)")
        parser.add_argument("--mom_sgd_ph",           type=str,   default=default_args["mom_sgd_ph"],         help="SGD momentum for the privacy heads (comma-separated)")
        parser.add_argument("--wd_ph",                type=str,   default=default_args["wd_ph"],              help="Weight decay for the privacy heads (comma-separated)")
        parser.add_argument("--patience_ph",          type=str,   default=default_args["patience_ph"],        help="Patience for the privacy heads learning rate scheduler (comma-separated)")
        parser.add_argument("--threshold_sched_ph",   type=str,   default=default_args["threshold_sched_ph"], help="Threshold for the privacy heads learning rate scheduler (comma-separated)")
        parser.add_argument("--cooldown_ph",          type=str,   default=default_args["cooldown_ph"],        help="Cooldown for the privacy heads learning rate scheduler (comma-separated)")
        parser.add_argument("--sched_factor_ph",      type=str,   default=default_args["sched_factor_ph"],    help="Factor for the privacy heads learning rate scheduler (comma-separated)")
        parser.add_argument("--min_lr_ph",            type=str,   default=default_args["min_lr_ph"],          help="Minimum learning rate for the privacy heads (comma-separated)")
        parser.add_argument(
            "--gamma",
            type=str,
            required=True,
            help="Weight of private loss for each ph (comma-separated, e.g. --gamma 0.1,0.2,0.3,0.4)",
        )
        parser.add_argument(
            "--pruning_retraining",
            type=str,
            required=True,
            choices=["none", "retrain"],
            help=(
                "Strategia di riaddestramento durante la fase di pruning globale: "
                "'none' = nessun riaddestramento per ogni sparsity; "
                "'retrain' = per ogni sparsity: PH + backbone con minimizzazione MI."
            ),
        )
        parser.add_argument("--train_ph_each_epoch",  action='store_true', default=default_args["train_ph_each_epoch"], help="Train phs each epoch")
        parser.add_argument("--max_ph_deterioration", type=float, default=default_args["max_ph_deterioration"],help="Minimum factor of multiplication of the phs val loss before retraining them")
        parser.add_argument("--train_PHs_every_sparsity",action='store_true', default=default_args["train_PHs_every_sparsity"], help="Train phs at each sparsity level")
        # Model parameters
        parser.add_argument("--model",                type=str,   default=default_args["model"])
        parser.add_argument("--pretrained",           action='store_true', default=default_args["pretrained"])
        parser.add_argument("--reload_config",        type=str,   default=default_args["reload_config"])
        parser.add_argument("--save_model",           action='store_true', default=default_args["save_model"])
        parser.add_argument("--save_path",            type=str,   default=default_args["save_path"])
        parser.add_argument("--save_each_epoch",      action='store_true', default=default_args["save_each_epoch"])
        parser.add_argument("--save_only_best_models",action='store_true', default=default_args["save_only_best_models"],
                            help="Salva solo i checkpoint migliori evitando quelli intermedi")

        # Pruning parameters
        parser.add_argument("--prune",                action='store_true', default=default_args["prune"])
        parser.add_argument("--blocks_to_prune",      type=str,   default=default_args["blocks_to_prune"],    help="Blocks to prune (comma-separated, e.g. --blocks_to_prune 0,1,2,3)")
        parser.add_argument("--prune_block_by_block", action='store_true',  default=default_args["prune_block_by_block"])
        parser.add_argument("--pruning_method",       type=str,   default=default_args["pruning_method"])
        parser.add_argument("--pruning_amount",       type=str,   default=default_args["pruning_amount"],     help="Pruning amount for each block (comma-separated)")
        parser.add_argument("--final_sparsity",       type=str,   default=default_args["final_sparsity"],     help="Final sparsity for each block (comma-separated)")
        parser.add_argument("--criterion_best",       type=str,   default=default_args["criterion_best"])
        parser.add_argument("--max_deter_pruning",    type=float, default=default_args["max_deter_pruning"], help="Threshold for pruning: best metric should not lose more than this proportion")
        parser.add_argument("--max_loop_without_imp", type=int,   default=default_args["max_loop_without_imp"], help="Maximum number of loops without improvement before stopping pruning")
        parser.add_argument("--pruning_criterion",       type=str,   default=default_args["pruning_criterion"], help="Magnitude type for pruning", choices=["weight", "gradient"])
        parser.add_argument("--Nsparsities",             type=int,   default=default_args["Nsparsities"],      help="Number of sparsity candidates for the binary search")
        args = parser.parse_args()

        # Add defaults for any fields not exposed through argparse
        parser_arg_dests = {action.dest for action in parser._actions}
        for key, value in default_args.items():
            if key not in parser_arg_dests and not hasattr(args, key):
                setattr(args, key, value)

    # Convert string parameters to lists
    if not args.projectName:
        raise ValueError("--projectName must be provided")

    args.save_path = os.path.join(args.save_path, args.projectName)
    args.used_phs = str_to_list(args.used_phs)
    args.nb_epochs_ph = str_to_list(args.nb_epochs_ph)
    args.refresh_epochs_ph = str_to_list(args.refresh_epochs_ph)
    args.batch_size_ph = str_to_list(args.batch_size_ph)
    args.lr_ph = str_to_list(args.lr_ph, float)
    args.mom_sgd_ph = str_to_list(args.mom_sgd_ph, float)
    args.wd_ph = str_to_list(args.wd_ph, float)
    args.patience_ph = str_to_list(args.patience_ph)
    args.threshold_sched_ph = str_to_list(args.threshold_sched_ph, float)
    args.cooldown_ph = str_to_list(args.cooldown_ph)
    args.sched_factor_ph = str_to_list(args.sched_factor_ph, float)
    args.min_lr_ph = str_to_list(args.min_lr_ph, float)
    args.gamma = str_to_list(args.gamma, float)
    args.blocks_to_prune = str_to_list(args.blocks_to_prune)
    args.pruning_amount = str_to_list(args.pruning_amount, float)
    args.final_sparsity = str_to_list(args.final_sparsity, float)

    if args.minimize_MI:
        if args.gamma is None or (isinstance(args.gamma, list) and len(args.gamma) == 0):
            raise ValueError("--gamma must be provided explicitly when minimizing mutual information")
    else:
        args.gamma = [0.0 for _ in range(len(args.used_phs))]

    # Ensure all parameters have the same length as used_phs
    handle_all(args)

    if args.seed is None:
        raise ValueError("A seed must be provided explicitly (--seed on the CLI, or the seed entry in kwargs).")

    initialize_seeds(args)
    args.run_dir = create_save_dir(args, False)
    return args

def update_args(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    if args.train_private:
        args.used_phs = []
        args.patience_ph= None
        args.sched_factor_ph= None
        args.lr_ph= None
        args.nb_epochs_ph = 0
    if not args.minimize_MI:
        args.gamma = [0.0 for _ in range(len(args.used_phs))]
    else: 
        temp_epochs = args.nb_epochs_ph
        temp_batch = args.batch_size_ph
        temp_lr = args.lr_ph
        temp_mom = args.mom_sgd_ph
        temp_wd = args.wd_ph
        temp_patience = args.patience_ph
        temp_factor = args.sched_factor_ph
        temp_gamma = args.gamma

        args.nb_epochs_ph = {}
        args.batch_size_ph = {}
        args.lr_ph = {}
        args.mom_sgd_ph = {}
        args.wd_ph = {}
        args.patience_ph = {}
        args.sched_factor_ph = {}
        args.gamma = {}

        for ph in args.used_phs:
            i = ph
            args.nb_epochs_ph[ph] = temp_epochs[i]
            args.batch_size_ph[ph] = temp_batch[i]
            args.lr_ph[ph] = temp_lr[i]
            args.mom_sgd_ph[ph] = temp_mom[i]
            args.wd_ph[ph] = temp_wd[i]
            args.patience_ph[ph] = temp_patience[i]
            args.sched_factor_ph[ph] = temp_factor[i]
            args.gamma[ph] = temp_gamma[i]
    if not args.prune:
        args.pruning_method = None
        args.pruning_amount_per_training = 0
        args.blocks_to_prune = []
        args.prune_block_by_block = False
        args.final_sparsity = 0
    elif not args.prune_block_by_block:
        args.final_sparsity = args.final_sparsity[0]
        args.pruning_amount = args.pruning_amount[0]

def print_args_summary(args):
    print("Arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("\n")
    return


def initialize_seeds(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    return
