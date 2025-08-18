import argparse

import torch
from time_series_baseframe import base_time_series
import pandas as pd
import os
import models
from utils.logger import setup_experiment_logging

def run_experiments(args):
    results_file = os.path.join(args.output_dir, "gca_GT_NPDC_market.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output directory created")

    gca = base_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
                          args.generators, args.discriminators,
                          args.ckpt_dir, args.output_dir,
                          args.window_sizes,
                          ckpt_path=args.ckpt_path,
                          initial_learning_rate=args.lr,
                          train_split=args.train_split,
                          do_distill_epochs=args.distill_epochs,
                          cross_finetune_epochs=args.cross_finetune_epochs,
                          device=args.device,
                          seed=args.random_seed)

    for target in args.target_columns:
        # for target,feature in zip(target_columns,feature_columns):
        target_feature_columns = args.feature_columns
        # target_feature_columns = feature_columns
        # target_feature_columns=target_feature_columns.extend(target)
        target_feature_columns.extend(target)
        # target_feature_columns.append(target)
        print("using features:", target_feature_columns)

        gca.process_data(args.data_path,args.start_timestamp, args.end_timestamp, target, target_feature_columns)
        gca.init_dataloader()
        gca.init_model(args.num_classes)


        logger = setup_experiment_logging(args.output_dir, vars(args))

        if args.mode == "train":
            results, best_model_state = gca.train(logger)
            torch.save(best_model_state, args.ckpt_dir)
        elif args.mode == "pred":
            results = gca.pred()

        result_row = {
            "feature_columns": args.feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "train_mse_per_target": results["train_mse_per_target"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"],
            "test_mse_per_target": results["test_mse_per_target"]
        }
        df = pd.DataFrame([result_row])
        df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)


if __name__ == "__main__":
    print("============= Available models ==================")
    for name in dir(models):
        obj = getattr(models, name)
        if isinstance(obj, type):
            print("\t", name)
    print("** Any other models please refer to add you model name to models.__init__ and import your costumed ones.")
    print("===============================================\n")

    parser = argparse.ArgumentParser(description="Run experiments for triple GAN model")
    parser.add_argument('--notes', type=str, required=False, help="Leave your setting in this note",
                        default="")
    parser.add_argument('--data_path', type=str, required=False, help="Path to the input data file",
                        default="database/processed_PTA_day.csv")
    parser.add_argument('--output_dir', type=str, required=False, help="Directory to save the output",
                        default=r"..\main/out_put/multi")
    parser.add_argument('--ckpt_dir', type=str, required=False, help="Directory to save the checkpoints",
                        default="ckpt")
    parser.add_argument('--feature_columns', type=list, help="Window size for first dimension", default=list(range(2,19)))
    # parser.add_argument('--feature_columns', type=list, help="Window size for first dimension", default=[])
    # parser.add_argument('--feature_columns', type=list, help="Window size for first dimension", default=list(range(2,24)))
    parser.add_argument('--target_columns', type=list, help="Window size for first dimension", default=[list(range(1, 2))])
    parser.add_argument('--start_timestamp', type=int, help="start row", default=31)
    parser.add_argument('--end_timestamp', type=int, help="end row", default=-1)
    # parser.add_argument('--start_timestamp', type=int, help="start row", default=1)
    # parser.add_argument('--end_timestamp', type=int, help="end row", default=2400)
    parser.add_argument('--window_sizes', nargs='+', type=int, help="Window size for first dimension", default=[15]) #, 10, 15
    parser.add_argument('--N_pairs', "-n", type=int, help="numbers of generators etc.", default=1)
    parser.add_argument('--num_classes', "-n_cls", type=int, help="numbers of class in classifier head, e.g. 0 par/1 rise/2 fall", default=3)
    parser.add_argument('--generators', "-gens", nargs='+', type=str, help="names of generators",
                        default=["transformer"]) # , "lstm", "transformer"
    parser.add_argument('--discriminators', "-discs", type=list, help="Window size for first dimension", default=None)
    parser.add_argument('--distill_epochs', type=int, help="Whether to do distillation", default=0)
    parser.add_argument('--cross_finetune_epochs', type=int, help="Whether to do distillation", default=0)
    parser.add_argument('--device', type=list, help="Device sets", default=[0])

    parser.add_argument('--num_epochs', type=int, help="epoch", default=9999)
    parser.add_argument('--lr', type=int, help="initial learning rate", default=2e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="Train-test split ratio", default=0.7)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",  # 'float16', 'bfloat16', 'none'
        choices=["float16", "bfloat16", "none"],
        help="AMP：float16, bfloat16, or none"
    )
    parser.add_argument('--mode', type=str, choices=["pred", "train"],
                        help="If train, it will also pred, while it predicts, it will laod the model checkpoint saved before.",
                        default="train")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint path", default="lastest")

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    run_experiments(args)
