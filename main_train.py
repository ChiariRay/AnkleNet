from utils.utils import seed_everything

seed_everything(1111)

import torch
import warnings
import argparse
import pandas as pd
from functools import partial
from models import create_model
from models.trainer import Trainer, Callback
from models.metrics import compute_aucs, compute_acc, compute_se, compute_sp
from models.utils_model import CustomWarmupStaticDecayLR
from datasets import my_dataloder, create_transforms
from utils.utils_option import parse_yaml

warnings.filterwarnings("ignore")


def main(
    opt_path="config/experiment_opt/anklenet.yaml",
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt", type=str, default=opt_path, help="Path to option Yaml file."
    )
    parser.add_argument("--dist", default=False)

    opt = parse_yaml(parser.parse_args().opt)
    opt["dist"] = parser.parse_args().dist
    opt["num_classes"] = {"bilateral": 2}.get(opt["label"], 1)

    df = pd.read_csv(opt["data"]["devdata_path"])
    df_test = pd.read_csv(opt["data"]["testdata_path"])
    transform_img = create_transforms(
        num_slices=opt["transform"]["num_slices"],
        resize=opt["transform"]["resize"],
        img_size=opt["transform"]["img_size"],
    )
    train_loader, val_loader = my_dataloder(
        data=df,
        batch_size=opt["batch_size"],
        sample_list=opt["data"]["sample_list"],
        num_workers=opt["num_work"],
        transforms=transform_img,
    )
    test_loader = my_dataloder(
        data=df_test,
        test=True,
        batch_size=opt["batch_size"],
        num_workers=opt["num_work"],
        transforms=transform_img,
    )

    model = create_model(
        backbone_name=opt["model"]["backbone_name"],
        num_slices=opt["transform"]["num_slices"],
        img_size=opt["transform"]["img_size"],
        num_classes=opt["num_classes"],
        num_embeddings=opt["model"]["num_embeddings"],
        dropout_rate=opt["model"]["dropout_rate"],
        anklenet_args=opt["model"]["anklenet"],
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=opt["lr"], weight_decay=opt["wd"]
    )
    lr_scheduler = CustomWarmupStaticDecayLR(
        optimizer=optimizer,
        epochs_warmup=5,  # warmup
        epochs_static=15,
        epochs_decay=1,
        decay_factor=0.9, # decay_factor
    )  

    auc_score = partial(compute_aucs, num_classes=opt["num_classes"], decimal_places=3)
    acc_score = partial(compute_acc, num_classes=opt["num_classes"], decimal_places=3)
    se_score = partial(compute_se, num_classes=opt["num_classes"], decimal_places=3)
    sp_score = partial(compute_sp, num_classes=opt["num_classes"], decimal_places=3)

    metrics = {
        # 'auc': roc_auc_score,
        # 'ap': average_precision_score,
        "auc": auc_score,
        # 'acc': acc_score,
        # 'se': se_score,
        # 'sp': sp_score,
    }

    trainer = Trainer(
        model=model,
        desc=opt["desc"],
        optimizer=optimizer,
        criterion=criterion,
        plane=opt["plane"],
        label=opt["label"],
    )

    train_callback = Callback(
        mode="train",
        metrics=metrics,
        monitor="auc",
        best_wts_num=2,
    )

    _, val_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opt["epoch"],
        callback=train_callback,
        lr_scheduler=lr_scheduler,
        device=opt["device"],
        save_weights=True,
    )

    print(val_results.to_string(index=False))

    print("Testing data ...")

    test_results = trainer.test(
        test_loader=test_loader, metrics=metrics, device=opt["device"]
    )

    test_log = {}
    test_log["test auc_avg"] = test_results.iloc[0]["test avg_auc"]
    if isinstance(test_results.iloc[0]["test auc"], list):
        for i, auc in enumerate(test_results.iloc[0]["test auc"], start=1):
            test_log[f"test auc_{i}"] = auc
    print(test_results.to_string(index=False))



if __name__ == "__main__":
    main(
        opt_path="config/experiment_opt/anklenet.yaml",
        )
