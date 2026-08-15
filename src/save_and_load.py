import os
import time
import numpy as np
import pandas as pd
import torch

def create_save_dir(args, create=False):
    if create:
        os.makedirs(args.save_path, exist_ok=True)

        runs_file = os.path.join(args.save_path, "runs.csv")

        args_dict = vars(args)
        columns = list(args_dict.keys()) + ['timestamp", "run_dir']

        if os.path.exists(runs_file):
            runs_df = pd.read_csv(runs_file)
        else:
            runs_df = pd.DataFrame(columns=columns)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        while True:
            run_dir = "".join(np.random.choice(list("0123456789abcdef"), 16))
            if not os.path.exists(runs_file) or run_dir not in runs_df['run_dir'].values:
                break

        new_run = {**args_dict, "timestamp": timestamp, "run_dir": run_dir}
        new_run_df = pd.DataFrame([new_run])

        runs_df = pd.concat([runs_df, new_run_df], ignore_index=True)
        runs_df.to_csv(runs_file, index=False)

        os.path.join(args.save_path, run_dir)
        os.makedirs(os.path.join(args.save_path, run_dir), exist_ok=True)
    else : run_dir = "None"
    return run_dir

def save_model_checkpoints(args, main_model, privacy_heads, epoch, sparsity= 0, init=False):
    
    if type(sparsity) == dict:
        sparsity = f"global{sparsity['global']}_block{'-'.join([f'{k}_{v}' for k,v in sparsity['layer'].items()])}"
    if init:
        model_save_dir = os.path.join(args.save_path, args.run_dir, f"init")
    else:
        epoch_log = f"epoch_glob{epoch['global']['main_clf']}"
        for i in args.used_phs:
            epoch_log += f"_ph{i}_{epoch['global'][f'ph{i}']}"
        model_save_dir = os.path.join(args.save_path, args.run_dir, f"{sparsity}", epoch_log)
    model_save_dir = model_save_dir.replace('.', 'p')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(main_model.state_dict(), os.path.join(model_save_dir, "model.pth"))

    for i in args.used_phs:
        torch.save(privacy_heads[i].state_dict(),
                   os.path.join(model_save_dir, f"privacy_head_{i}.pth"))

def load_previous_model(args, main_model, privacy_heads, warmup=False):
    path_to_models = os.path.join(args.save_path, args.reload_config)
    checkpoint = torch.load(f"{path_to_models}/model.pth")
    main_model.load_state_dict(checkpoint)
    for i in args.used_phs:
        checkpoint = torch.load(f"{path_to_models}/privacy_head_{i}.pth")
        privacy_heads[i].load_state_dict(checkpoint)
    return main_model, privacy_heads
