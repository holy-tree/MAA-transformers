import argparse
from time_series_maa import MAA_time_series
import pandas as pd
import os
import models
from utils.logger import setup_experiment_logging
import logging

# 更新 name_map，添加新的合约代码和对应的标准名称
name_map = {
    "rb9999": "Rebar",
    "i9999": "Iron Ore",
    "cu9999": "Copper",
    "ni9999": "Nickel",
    "sc9999": "Crude Oil",
    "pg9999": "LPG",
    "y9999": "Soybean Oil",
    "ag9999": "Silver",
    "m9999": "Soybean Meal",
    "c9999": "Corn",
    "TA9999": "PTA",
    "UR9999": "Urea",
    "OI9999": "Rapeseed Oil",
    "au9999": "Gold",
    "IH9999": "SSE 50 Index Futures",
    "T9999": "10-Year Treasury Futures",
    "CF9999": "Cotton",
    "AP9999": "Apple Futures",
    "IF9999": "CSI300",
    "IC9999": "CSI500",
    "ts9999": "2-Year Treasury Futures",
}

model_name_map = {
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer"
}


def run_experiments(args):
    # 创建保存结果的CSV文件
    if args.N_pairs > 1:
        results_file = os.path.join('./results/output', "maa_results.csv")
    else:
        if not args.ERM:
            results_file = os.path.join('./results/output', "gan_results.csv")
        else:
            results_file = os.path.join('./results/output', "erm_results.csv")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output directory created")

    filename = os.path.basename(args.data_path)

    # 修正文件名处理逻辑以适应新的命名格式
    # 假设文件名格式是 {symbols}_{freq}_data.csv，所以我们取第一个 '_' 之前的部分
    raw_target_name = filename.split('_')[0]

    args.target_name = name_map.get(raw_target_name, raw_target_name)

    maa = MAA_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
                          args.generator_names, args.discriminators,
                          args.output_dir,
                          args.window_sizes,
                          args.ERM, args.target_name,
                          ckpt_path=args.ckpt_path,
                          initial_learning_rate=args.lr,
                          train_split=args.train_split,
                          do_distill_epochs=args.distill_epochs,
                          cross_finetune_epochs=args.cross_finetune_epochs,
                          device=args.device,
                          seed=args.random_seed)

    target_columns = args.target_columns
    # 运行实验，获取结果
    target_feature_columns = args.feature_columns

    target_feature_columns = list(zip(target_feature_columns[::2], target_feature_columns[1::2]))
    target_feature_columns = [list(range(a, b)) for (a, b) in target_feature_columns]

    # target_feature_columns.append(target)
    print("using features:", target_feature_columns)

    maa.process_data(args.data_path, args.start_timestamp, args.end_timestamp, target_columns, target_feature_columns)
    maa.init_dataloader()
    maa.init_model(args.num_classes)
    # 定义映射关系（全部小写 -> 标准形式）

    # 替换 self.generator_names 中的名称
    maa.generator_names = [
        model_name_map.get(name.lower(), name)  # 忽略大小写，找不到就保留原名
        for name in maa.generator_names
    ]

    logger = setup_experiment_logging(args.output_dir, vars(args))

    if args.mode == "train":
        results_acc = maa.train(logger)
        train_results, test_results = maa.pred()
        for i, result_acc in enumerate(results_acc):
            test_results[i]["Accuracy"] = round(results_acc[i] * 100, 2)
            test_results[i]["target"] = args.target_name

    elif args.mode == "pred":
        results = maa.pred()

    # 将结果保存到CSV

    df = pd.DataFrame(test_results)
    df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)


if __name__ == "__main__":
    print("============= Available models ==================")
    for name in dir(models):
        obj = getattr(models, name)
        if isinstance(obj, type):
            print("\t", name)
    print("** Any other models please refer to add you model name to models.__init__ and import your costumed ones.")
    print("===============================================\n")

    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Run experiments for triple GAN model")
    parser.add_argument('--notes', type=str, required=False, help="Leave your setting in this note")

    # 修改这里：添加 --symbol 和 --freq 参数，并移除 --data_path
    parser.add_argument('--symbol', type=str, required=False,
                        help="The symbol of the data to be processed, e.g., 'rb9999'")
    parser.add_argument('--freq', type=str, choices=['daily', '1min'], required=False,
                        help="Data frequency: 'daily' or '1min'")
    parser.add_argument('--data_path', type=str, required=False,
                        help="Path to the input data file. This argument is an alternative to --symbol and --freq")

    parser.add_argument('--output_dir', type=str, required=False, help="Directory to save the output")

    parser.add_argument('--ERM', type=bool, help="whether to do use discriminator", default=False)
    parser.add_argument('--feature_columns', nargs='+', type=int, help="features choosed to be used as input",
                        default=[1,5,1,5,1,5])

    parser.add_argument('--target_columns', nargs='+', type=int, help="target to be predicted", default=[1,2,3,4])
    parser.add_argument('--start_timestamp', type=int, help="start row", default=31)
    parser.add_argument('--end_timestamp', type=int, help="end row", default=-1)

    parser.add_argument('--window_sizes', nargs='+', type=int, help="Window size for first dimension",
                        default=[5, 10, 15])
    parser.add_argument('--N_pairs', "-n", type=int, help="numbers of generators etc.", default=3)
    parser.add_argument('--num_classes', "-n_cls", type=int,
                        help="numbers of class in classifier head, e.g. 0 par/1 rise/2 fall", default=3)
    parser.add_argument('--generator_names', "-gens", nargs='+', type=str, help="names of generators",
                        default=["gru", "lstm", "transformer"])
    # default=["lstm"])
    parser.add_argument('--discriminators', "-discs", type=list, help="names of discriminators", default=None)
    parser.add_argument('--distill_epochs', type=int, help="Epochs to do distillation", default=1)
    parser.add_argument('--cross_finetune_epochs', type=int, help="Epochs to do distillation", default=5)
    parser.add_argument('--device', type=list, help="Device sets", default=[0])

    parser.add_argument('--num_epochs', type=int, help="epoch", default=1)
    parser.add_argument('--lr', type=int, help="initial learning rate", default=2e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="Train-test split ratio", default=0.7)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",  # 可选：'float16', 'bfloat16', 'none'
        choices=["float16", "bfloat16", "none"],
        help="自动混合精度类型（AMP）：float16, bfloat16, 或 none（禁用）"
    )
    parser.add_argument('--mode', type=str, choices=["pred", "train"],
                        help="If train, it will also pred, while it predicts, it will laod the model checkpoint saved before.",
                        default="train")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint path", default="latest")

    args = parser.parse_args()

    # 动态构建 data_path 和 output_dir
    if args.symbol and args.freq:
        args.data_path = f"../data/raw_data/{args.symbol}_{args.freq}_data.csv"
        args.output_dir = f"../results/output/{args.symbol}_{args.freq}_data"

    # 检查 data_path 是否存在
    if not args.data_path:
        parser.error("You must specify either --data_path or both --symbol and --freq.")

    if not os.path.exists(args.data_path):
        parser.error(f"Error: Data file not found at {args.data_path}")

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    # 调用run_experiments函数
    run_experiments(args)