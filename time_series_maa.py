from argparse import Namespace

import joblib

from MAA_base import MAABase
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from utils.multiGAN_trainer_disccls import train_multi_gan
from utils.baseframe_trainer import train_baseframe

from typing import List, Optional
import models
import time
import glob
from utils.evaluate_visualization import *


def log_execution_time(func):
    """装饰器：记录函数的运行时间，并动态获取函数名"""

    @wraps(func)  # 保留原函数的元信息（如 __name__）
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 动态获取函数名（支持类方法和普通函数）
        func_name = func.__name__
        print(f"MAA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


def generate_labels(y):
    """
    根据每个时间步 y 是否比前一时刻更高，生成三分类标签：
      - 2: 当前值 > 前一时刻（上升）
      - 0: 当前值 < 前一时刻（下降）
      - 1: 当前值 == 前一时刻（平稳）
    对于第一个时间步，默认赋值为1（平稳）。

    参数：
        y: 数组，形状为 (样本数, ) 或 (样本数, 1)
    返回：
        labels: 生成的标签数组，长度与 y 相同
    """
    y = np.array(y).flatten()  # 转成一维数组
    labels = [0]  # 对于第一个样本，默认平稳
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class MAA_time_series(MAABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generator_names: List, discriminators_names: Optional[List],
                 output_dir: str,
                 window_sizes: int,
                 ERM: bool,
                 target_name: str,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generator_names: list object，包括了表示具有不同特征的生成器的名称
        :param discriminators_names: list object，包括了表示具有不同判别器的名称，如果没有就不写默认一致
        :param output_path: 可视化、损失函数的log等输出目录
        :param ckpt_path: 预测时保存的检查点
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generator_names, discriminators_names,
                         ckpt_path,
                         output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed
                         )  # 调用父类初始化

        self.args = args
        self.window_sizes = window_sizes
        self.ERM = ERM
        self.ckpt_dir = os.path.join(output_dir, 'ckpt')
        self.target_name = target_name
        # 初始化空字典
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        # 遍历 model 模块下的所有属性
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, data_path, start_row, end_row, target_columns, feature_columns_list):
        """
        Process the input data by loading, splitting, and normalizing it.

        Args:
            data_path (str): Path to the CSV data file
            target_columns (list): Indices of target columns
            feature_columns (list): Indices of feature columns

        Returns:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """
        print(f"Processing data with seed: {self.seed}")  # Using self.seed

        # Load data
        data = pd.read_csv(data_path)

        # Select target columns
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)

        # # Select feature columns
        # x = data.iloc[start_row:end_row, feature_columns].values
        # feature_column_names = data.columns[feature_columns]
        # print("Feature columns:", feature_column_names)

        # Process each set of feature columns
        x_list = []
        feature_column_names_list = []
        self.x_scalers = []  # Store multiple x scalers

        for feature_columns in feature_columns_list:
            # Select feature columns
            x = data.iloc[start_row:end_row, feature_columns].values
            feature_column_names = data.columns[feature_columns]
            print("Feature columns:", feature_column_names)

            x_list.append(x)
            feature_column_names_list.append(feature_column_names)

        # —— 1. 计算并打印总体 y 的均值和方差 ——
        print(f"Overall  Y mean: {y.mean():.4f}, var: {y.var():.4f}")

        # Data splitting using self.train_split
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        # train_x, test_x = x[:train_size], x[train_size:]
        # Split each x in the list
        train_x_list = [x[:train_size] for x in x_list]
        test_x_list = [x[train_size:] for x in x_list]
        train_y, test_y = y[:train_size], y[train_size:]

        # —— 3. 计算并打印 train 和 test 的均值、方差 ——
        print(f"Train    Y mean: {train_y.mean():.4f}, var: {train_y.var():.4f}")
        print(f"Test     Y mean: {test_y.mean():.4f}, var: {test_y.var():.4f}")

        # Normalize each x set separately
        self.train_x_list = []
        self.test_x_list = []
        for train_x, test_x in zip(train_x_list, test_x_list):
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            self.train_x_list.append(x_scaler.fit_transform(train_x))
            self.test_x_list.append(x_scaler.transform(test_x))
            self.x_scalers.append(x_scaler)  # Store all x scalers

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        # self.train_x = self.x_scaler.fit_transform(train_x)
        # self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        # 生成训练集的分类标签（直接在 GPU 上生成）
        self.train_labels = generate_labels(self.train_y)
        # 生成测试集的分类标签
        self.test_labels = generate_labels(self.test_y)
        print(self.train_y[:5])
        print(self.train_labels[:5])
        # ------------------------------------------------------------------

    def create_sequences_combine(self, x_list, y, label, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []
        # Create sequences for each x in x_list
        for x in x_list:
            x_seq = []
            for i in range(start, x.shape[0]):
                tmp_x = x[i - window_size: i, :]
                x_seq.append(tmp_x)
            x_.append(np.array(x_seq))

        # Combine x sequences along feature dimension
        x_ = np.concatenate(x_, axis=-1)

        for i in range(start, y.shape[0]):
            # tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            # x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
            label_gan.append(tmp_label_gan)

        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        label_gan = torch.from_numpy(np.array(label_gan)).float()
        return x_, y_, y_gan, label_gan

    @log_execution_time
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""

        # Sliding Window Processing
        # 分别生成不同 window_size 的序列数据
        train_data_list = [
            self.create_sequences_combine(self.train_x_list, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x_list, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # 分别提取 x、y、y_gan 并堆叠
        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  不同 window_size 会导致 W 不一样，只能在 W 相同时用 stack
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []

        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # 最后一个设置为 shuffle=True，其余为 False
            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True  # 丢弃最后一个不足 batch size 的数据
            )
            self.dataloaders.append(dataloader)

    def init_model(self, num_cls):
        """模型结构初始化"""
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            # 获取对应的 x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]
            print(f"Generator {i + 1}: Input shape x={x.shape}, Output shape y={y.shape}")

            # 初始化生成器
            GenClass = self.generator_dict[name]
            if "itransformer" in name:
                itransformer_configs = Namespace(
                    seq_len=self.window_sizes[i],
                    pred_len=self.args.output_len,  # 需要在run_multi_gan.py中添加
                    d_model=self.args.d_model,
                    embed=self.args.embed,
                    freq=self.args.fre_itrans,
                    dropout=self.args.dropout,
                    n_heads=self.args.n_heads,
                    d_ff=self.args.d_ff,
                    activation=self.args.activation,
                    e_layers=self.args.e_layers,
                    factor=self.args.factor,
                    output_attention=self.args.output_attention,
                    use_norm=self.args.use_norm,
                    class_strategy=self.args.class_strategy,
                    num_classes=self.args.num_classes,
                )
                gen_model = GenClass(itransformer_configs).to(self.device)
            elif name == 'ptransformer':
                # print(end_timestamp - start_timestamp +1)
                window_size = self.window_sizes[i]
                gen_model = GenClass(input_dim=x.shape[-1], seq_len=window_size).to(self.device)
            elif "fits" in name:
                # 硬编码所有 FITS 模型的配置
                current_seq_len = self.window_sizes[i]
                calculated_cut_freq = current_seq_len // 2+1
                fits_configs = Namespace(
                    seq_len=self.window_sizes[i],  # 例如，硬编码输入序列长度
                    pred_len=self.args.output_len,  # 例如，硬编码预测长度
                    enc_in=12,  # 硬编码为 OHLC 的4个通道
                    d_model=512,  # 例如，硬编码模型维度
                    embed='fixed',  # 例如，硬编码嵌入类型
                    freq='m',  # 例如，硬编码频率
                    dropout=0.05,  # 例如，硬编码 dropout
                    n_heads=8,  # 例如，硬编码注意力头数
                    d_ff=2048,  # 例如，硬编码前馈网络维度
                    activation='gelu',  # 例如，硬编码激活函数
                    e_layers=2,  # 例如，硬编码编码器层数
                    factor=1,  # 例如，硬编码 factor
                    output_attention=False,  # 例如，硬编码是否输出注意力
                    individual=False,
                    use_norm=True,  # 例如，硬编码是否使用归一化
                    class_strategy='prob',  # 例如，硬编码分类策略
                    num_classes=3,  # 硬编码分类类别数量
                    cut_freq=calculated_cut_freq,  # 硬编码截止频率
                )
                gen_model = GenClass(fits_configs).to(self.device)
            elif "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=4).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], 4).to(self.device)

            self.generators.append(gen_model)

            # 初始化判别器（默认只用 Discriminator3）
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1], num_cls=num_cls).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """初始化训练所需的超参数"""
        # 初始化：对角线上为1，其余为0，最后一列为1.0
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # 最后一列为 scale
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            # 最终：均分组合，最后一列为1.0
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass

        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self, logger):
        if not self.ERM:
            best_acc, best_model_state = train_multi_gan(
                self.args, self.generators, self.discriminators, self.dataloaders,
                self.window_sizes,
                self.train_x_all, self.train_y_all, self.test_x_all,
                self.test_y_all, self.test_label_gan_all,
                self.do_distill_epochs, self.cross_finetune_epochs,
                self.num_epochs,
                self.output_dir,
                self.device,
                init_GDweight=self.init_GDweight,
                final_GDweight=self.final_GDweight,
                logger=logger)
        else:
            best_acc, best_model_state = train_baseframe(
                self.generators[0], self.dataloaders[0],
                self.y_scaler,
                self.train_x_all[0], self.train_y_all,
                self.test_x_all[0], self.test_y_all,
                self.train_label_gan_all[0], self.test_label_gan_all[0],
                self.num_epochs,
                self.output_dir,
                self.device,
                logger=logger)

        self.save_models(best_model_state)
        return best_acc

    def save_models(self, best_model_state):
        """
        保存所有 generator 和 discriminator 的模型参数，包含时间戳、模型名称或编号。
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        scaler_dir = os.path.join(ckpt_dir, "scalers")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)

        # 加载模型并设为 eval
        for i in range(self.N):
            self.generators[i].load_state_dict(best_model_state[i])
            self.generators[i].eval()

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            torch.save(gen.state_dict(), save_path)

        for i, disc in enumerate(self.discriminators):
            disc_name = type(disc).__name__
            save_path = os.path.join(disc_dir, f"{i + 1}_{disc_name}.pt")
            torch.save(disc.state_dict(), save_path)

        # 新增：保存归一化器
        # 保存 y_scaler
        joblib.dump(self.y_scaler, os.path.join(scaler_dir, "y_scaler.joblib"))
        # 保存 x_scalers 列表
        for i, x_scaler in enumerate(self.x_scalers):
            joblib.dump(x_scaler, os.path.join(scaler_dir, f"x_scaler_{i + 1}.joblib"))
        print("All models saved with timestamp and identifier.")

    def get_latest_ckpt_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(self.ckpt_dir)
        all_subdirs = [d for d in glob.glob(os.path.join(self.ckpt_dir, timestamp[0] + "*")) if os.path.isdir(d)]
        if not all_subdirs:
            raise FileNotFoundError("❌ No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"📂 Auto loaded checkpoint file: {latest}")
        return latest

    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"✅ Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"❌ Generator checkpoint not found at: {gen_path}")

    # def pred(self):
    #     if self.ckpt_path == "latest":
    #         self.ckpt_path = self.get_latest_ckpt_folder()

    #     print("Start predicting with all generators..")
    #     best_model_state = [None for _ in range(self.N)]
    #     current_path = os.path.join(self.ckpt_path, "generators")

    #     for i, gen in enumerate(self.generators):
    #         gen_name = type(gen).__name__
    #         save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
    #         state_dict = torch.load(save_path, map_location=self.device, weights_only=True)
    #         gen.load_state_dict(state_dict)
    #         best_model_state[i] = state_dict

    #     csv_save_dir = self.output_dir
    #     test_csv_dir = os.path.join(csv_save_dir, "test")
    #     if not os.path.exists(test_csv_dir):
    #         os.makedirs(test_csv_dir)
    #     train_csv_dir = os.path.join(csv_save_dir, "train")
    #     if not os.path.exists(train_csv_dir):
    #         os.makedirs(train_csv_dir)
    #     # —— 新增：遍历每个 generator，把“归一化后”->“原始价格”的真实/预测值保存到 CSV ——
    #     with torch.no_grad():
    #         for i, gen in enumerate(self.generators):
    #             gen.eval()
    #             # 准备输入、真实 y
    #             x_test = self.test_x_all[i]  # Tensor on device, shape=(N, W, F)
    #             y_true_norm = self.test_y_all.cpu().numpy()  # shape=(N,)
    #             # 前向预测（归一化后）
    #             y_pred_norm = gen(x_test)[0].cpu().numpy().reshape(-1, 1)  # (N,1)
    #             # 反归一化回原始值
    #             y_true = self.y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    #             y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()

    #             df = pd.DataFrame({
    #                 'true': y_true,
    #                 'pred': y_pred
    #             })
    #             test_csv_path = os.path.join(test_csv_dir,f'{self.generator_names[i]}.csv')
    #             df.to_csv(test_csv_path, index=False)
    #             print(f"Saved true vs pred for generator {self.generator_names[i]} at: {test_csv_path}")

    #     test_csv_paths = glob.glob(os.path.join(test_csv_dir, '*.csv'))
    #     all_true_series, pred_series_list, pred_labels = read_and_collect_data(test_csv_paths)
    #     if self.N>1:
    #     # 绘制密度图
    #         plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
    #                      no_grid=True,mode='test',target_name=self.target_name)

    #     with torch.no_grad():
    #         for i, gen in enumerate(self.generators):
    #             gen.eval()
    #             # 准备输入、真实 y
    #             x_train = self.train_x_all[i]  # Tensor on device, shape=(N, W, F)
    #             y_true_norm = self.train_y_all.cpu().numpy()  # shape=(N,)
    #             # 前向预测（归一化后）
    #             y_pred_norm = gen(x_train)[0].cpu().numpy().reshape(-1, 1)  # (N,1)
    #             # 反归一化回原始值
    #             y_true = self.y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
    #             y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()

    #             df = pd.DataFrame({
    #                 'true': y_true,
    #                 'pred': y_pred
    #             })

    #             train_csv_path = os.path.join(train_csv_dir,f'{self.generator_names[i]}.csv')
    #             df.to_csv(train_csv_path, index=False)
    #             print(f"Saved true vs pred for generator {self.generator_names[i]} at: {train_csv_path}")

    #     train_csv_paths = glob.glob(os.path.join(train_csv_dir, '*.csv'))
    #     all_true_series, pred_series_list, pred_labels = read_and_collect_data(train_csv_paths)

    #     # 绘制密度图
    #     if self.N > 1:
    #         plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
    #                      no_grid=True,mode='train',target_name=self.target_name)

    #     results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
    #                                    self.test_x_all, self.test_y_all, self.y_scaler,
    #                                    self.output_dir,self.generator_names,self.target_name,self.ERM)
    #     return results

    def pred(self):
        if self.ckpt_path == "latest":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generators")
        current_scaler_path = os.path.join(self.ckpt_path, "scalers")

        self.y_scaler = joblib.load(os.path.join(current_scaler_path, "y_scaler.joblib"))
        self.x_scalers = []
        num_x_scalers = len(glob.glob(os.path.join(current_scaler_path, "x_scaler_*.joblib")))
        for i in range(num_x_scalers):
            x_scaler = joblib.load(os.path.join(current_scaler_path, f"x_scaler_{i + 1}.joblib"))
            self.x_scalers.append(x_scaler)

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            state_dict = torch.load(save_path, map_location=self.device, weights_only=True)
            gen.load_state_dict(state_dict)
            best_model_state[i] = state_dict

        csv_save_dir = self.output_dir
        test_csv_dir = os.path.join(csv_save_dir, "test")
        if not os.path.exists(test_csv_dir):
            os.makedirs(test_csv_dir)
        train_csv_dir = os.path.join(csv_save_dir, "train")
        if not os.path.exists(train_csv_dir):
            os.makedirs(train_csv_dir)

        # 预测并保存测试集结果
        with torch.no_grad():
            y_pred_norm_list = []
            for i, gen in enumerate(self.generators):
                gen.eval()
                x_test = self.test_x_all[i]

                # 确保预测结果是二维的 (batch_size, num_features)
                y_pred_norm = gen(x_test)[0].cpu().numpy()
                y_pred_norm_list.append(y_pred_norm)

            # 关键修改：对所有生成器的预测结果取平均，将维度从 (3, 543, 4) 压缩为 (543, 4)
            y_pred_norm_full = np.mean(np.stack(y_pred_norm_list, axis=0), axis=0)

            # 获取真实值（归一化后）
            y_true_norm_full = self.test_y_all.cpu().numpy()

            # 反归一化回原始值
            # 现在 y_pred_norm_full 的形状是 (543, 4)，可以被 y_scaler 正确处理
            y_pred_full = self.y_scaler.inverse_transform(y_pred_norm_full)
            y_true_full = self.y_scaler.inverse_transform(y_true_norm_full)

            # 将结果保存到单个 CSV 文件中
            df_test = pd.DataFrame(y_true_full, columns=[f'true_{i + 1}' for i in range(y_true_full.shape[1])])
            for i in range(y_pred_full.shape[1]):
                df_test[f'pred_{i + 1}'] = y_pred_full[:, i]

            test_csv_path = os.path.join(test_csv_dir, 'all_generators.csv')
            df_test.to_csv(test_csv_path, index=False)
            print(f"Saved true vs pred for all generators at: {test_csv_path}")

        # 绘制密度图（如果你仍然想为每个生成器绘制，需要修改这部分）
        test_csv_paths = glob.glob(os.path.join(test_csv_dir, '*.csv'))
        all_true_series, pred_series_list, pred_labels = read_and_collect_data(test_csv_paths)
        if self.N > 1:
            plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
                         no_grid=True, mode='test', target_name=self.target_name)

        # 预测并保存训练集结果 (逻辑与测试集相同)
        with torch.no_grad():
            y_pred_norm_list = []
            for i, gen in enumerate(self.generators):
                gen.eval()
                x_train = self.train_x_all[i]
                y_pred_norm = gen(x_train)[0].cpu().numpy()
                y_pred_norm_list.append(y_pred_norm)

            # ⚠️ 关键修改：将 np.concatenate 替换为 np.mean(np.stack(...))
            y_pred_norm_full = np.mean(np.stack(y_pred_norm_list, axis=0), axis=0)

            y_true_norm_full = self.train_y_all.cpu().numpy()

            y_pred_full = self.y_scaler.inverse_transform(y_pred_norm_full)
            y_true_full = self.y_scaler.inverse_transform(y_true_norm_full)

            df_train = pd.DataFrame(y_true_full, columns=[f'true_{i + 1}' for i in range(y_true_full.shape[1])])
            for i in range(y_pred_full.shape[1]):
                df_train[f'pred_{i + 1}'] = y_pred_full[:, i]

            train_csv_path = os.path.join(train_csv_dir, 'all_generators.csv')
            df_train.to_csv(train_csv_path, index=False)
            print(f"Saved true vs pred for all generators at: {train_csv_path}")

        train_csv_paths = glob.glob(os.path.join(train_csv_dir, '*.csv'))
        all_true_series, pred_series_list, pred_labels = read_and_collect_data(train_csv_paths)

        if self.N > 1:
            plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
                         no_grid=True, mode='train', target_name=self.target_name)

        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir, self.generator_names, self.target_name, self.ERM)
        return results

    def distill(self):
        """评估模型性能并可视化结果"""
        pass

    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
