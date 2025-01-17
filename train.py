import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import SuDORMRFImprovedNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data import FUSSDataset
from data_utils.hrs_dataset import HSRDataset
# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--train_csv", default="train.csv", help="Full path to train csv file")
parser.add_argument("--val_csv", default="valid.csv", help="Full path to val csv file")
parser.add_argument("--mixture_train", default="mixture_train", help="Full path to mixture train folder")
parser.add_argument("--mixture_val", default="mixture_valid", help="Full path to mixture val folder")
parser.add_argument("--source_file_dir", default="concated_fg/machine", help="Full path to source file folder")
parser.add_argument("--bg_file_dir", default="concated_bg/env", help="Full path to background file folder")


def main(args, conf):
    train_set = HSRDataset(file_list_path=args.train_csv,
                           mixture_file_dir=args.mixture_train,
                           source_file_dir=args.source_file_dir,
                           bg_file_dir=args.bg_file_dir,
                           return_bg=False)
    val_set = HSRDataset(file_list_path=args.val_csv,
                         mixture_file_dir=args.mixture_val,
                         source_file_dir=args.source_file_dir,
                         bg_file_dir=args.bg_file_dir,
                         return_bg=False)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = SuDORMRFImprovedNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/config.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(plain_args, arg_dic)