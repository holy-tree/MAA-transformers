import torch
import torch.nn as nn
import copy

from utils.evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import logging  # NEW
from utils.util import get_autocast_context
from torch.cuda.amp import GradScaler

scaler = torch.amp.GradScaler('cuda')

def train_multi_gan(args,
                    generators, discriminators,
                    dataloaders,
                    window_sizes,
                    train_xes, train_y,
                    val_xes, val_y, val_labels,
                    distill_epochs, cross_finetune_epochs,
                    num_epochs,
                    output_dir,
                    device,
                    init_GDweight=[
                        [1, 0, 0, 1.0],  # alphas_init
                        [0, 1, 0, 1.0],  # betas_init
                        [0., 0, 1, 1.0]  # gammas_init...
                    ],
                    final_GDweight=[
                        [0.333, 0.333, 0.333, 1.0],  # alphas_final
                        [0.333, 0.333, 0.333, 1.0],  # betas_final
                        [0.333, 0.333, 0.333, 1.0]  # gammas_final...,
                    ],
                    logger=None,
                    dynamic_weight=False):
    N = len(generators)

    assert N == len(discriminators)
    assert N == len(window_sizes)
    assert N >= 1

    g_learning_rate = 2e-5
    d_learning_rate = 2e-5

    # 二元交叉熵【损失函数，可能会有问题
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizers_G = [torch.optim.AdamW(model.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    for model in generators]

    # 为每个优化器设置 ReduceLROnPlateau 调度器
    schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, min_lr=1e-7)
                  for optimizer in optimizers_G]

    optimizers_D = [torch.optim.Adam(model.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
                    for model in discriminators]

    best_epoch_mse = [-1 for _ in range(N)]
    best_epoch_acc = [-1 for _ in range(N)]

    # 定义生成历史记录的关键字
    """
    以三个为例，keys长得是这样得的：
    ['G1', 'G2', 'G3', 
    'D1', 'D2', 'D3', 
    'MSE_G1', 'MSE_G2', 'MSE_G3', 
    'val_G1', 'val_G2', 'val_G3', 
    'D1_G1', 'D2_G1', 'D3_G1', 'D1_G2', 'D2_G2', 'D3_G2', 'D1_G3', 'D2_G3', 'D3_G3'
    ]
    """

    keys = []
    g_keys = [f'G{i}' for i in range(1, N + 1)]
    d_keys = [f'D{i}' for i in range(1, N + 1)]
    MSE_g_keys = [f'MSE_G{i}' for i in range(1, N + 1)]
    val_loss_keys = [f'val_G{i}' for i in range(1, N + 1)]
    acc_keys = [f'acc_G{i}' for i in range(1, N + 1)]

    keys.extend(g_keys)
    keys.extend(d_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)
    keys.extend(acc_keys)

    d_g_keys = []
    for g_key in g_keys:
        for d_key in d_keys:
            d_g_keys.append(d_key + "_" + g_key)
    keys.extend(d_g_keys)

    # 创建包含每个值为np.zeros(num_epochs)的字典
    hists_dict = {key: np.zeros(num_epochs) for key in keys}

    best_mse = [float('inf') for _ in range(N)]
    best_acc = [0.0 for _ in range(N)]

    best_model_state = [None for _ in range(N)]

    patience_counter = 0
    patience = 15
    feature_num = train_xes[0].shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    if torch.cuda.is_available():
        print("GPU is available. The current device is:", torch.cuda.get_device_name(0))
    else:
        print("GPU is not available. Using CPU.")

    for epoch in range(num_epochs):
        epo_start = time.time()

        if epoch < 10:
            weight_matrix = torch.tensor(init_GDweight).to(device)
        elif dynamic_weight:
            # —— 动态计算 G-D weight 矩阵 ——
            # 从上一轮的 validation loss 里拿到每个 G 的损失
            # val_loss_keys = ['val_G1', 'val_G2', ..., 'val_GN']
            losses = torch.stack([
                torch.tensor(hists_dict[val_loss_keys[i]][epoch - 1])
                for i in range(N)
            ]).to(device)  # shape: [N]

            # 性能 Perf_i = -loss_i，beta 控制“硬度”
            perf = torch.exp(-losses)  # shape: [N]
            probs = perf / perf.sum()  # shape: [N], softmax over generators

            # 构造训练 Generator 时用的 N×N 矩阵：每行都是同一分布
            weight_G = probs.unsqueeze(0).repeat(N, 1)  # shape: [N, N]
            weight_G = weight_G + torch.eye(N, device=device)

            # 构造训练 Discriminator 时的 N×(N+1) 矩阵：最后一列保持 1.0（给真数据）
            ones = torch.ones((N, 1), device=device)
            weight_matrix = torch.cat([weight_G, ones], dim=1)  # shape: [N, N+1]
        else:
            weight_matrix = torch.tensor(final_GDweight).to(device)

        keys = []
        keys.extend(g_keys)
        keys.extend(d_keys)
        keys.extend(MSE_g_keys)
        keys.extend(d_g_keys)

        loss_dict = {key: [] for key in keys}

        # use the gap the equalize the length of different generators
        gaps = [window_sizes[-1] - window_sizes[i] for i in range(N - 1)]

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloaders[-1]):
            # TODO: maybe try to random select a gap from the whole time windows
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)
            # print(x_last.shape, y_last.shape, label_last.shape)

            X = []
            Y = []
            LABELS = []

            for gap in gaps:
                X.append(x_last[:, gap:, :])
                Y.append(y_last[:, gap:, :])
                LABELS.append(label_last[:, gap:, :].long())
            X.append(x_last.to(device))
            Y.append(y_last.to(device))
            LABELS.append(label_last.to(device).long())

            for i in range(N):
                generators[i].eval()
                discriminators[i].train()

            loss_D, lossD_G = discriminate_fake(args, X, Y, LABELS,
                                                generators, discriminators,
                                                window_sizes, target_num,
                                                criterion, weight_matrix,
                                                device, mode="train_D")

            # 3. 存入 loss_dict
            for i in range(N):
                loss_dict[d_keys[i]].append(loss_D[i].item())

            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    key = f'D{i}_G{j}'
                    loss_dict[key].append(lossD_G[i - 1, j - 1].item())

            # 根据批次的奇偶性交叉训练两个GAN
            # if batch_idx% 2 == 0:
            for optimizer_D in optimizers_D:
                optimizer_D.zero_grad()

            # TODO: to see whether there is need to add together

            scaler.scale(loss_D.sum(dim=0)).backward()

            for i in range(N):
                # optimizers_D[i].step()
                scaler.step(optimizers_D[i])
                scaler.update()

                discriminators[i].eval()
                generators[i].train()

            '''训练生成器'''
            weight = weight_matrix[:, :-1].clone().detach()  # [N, N]

            loss_G, loss_mse_G = discriminate_fake(args, X, Y, LABELS,
                                                   generators, discriminators,
                                                   window_sizes, target_num,
                                                   criterion, weight,
                                                   device,
                                                   mode="train_G")

            for i in range(N):
                loss_dict[g_keys[i]].append(loss_G[i].item())
                loss_dict["MSE_" + g_keys[i]].append(loss_mse_G[i].item())

            for optimizer_G in optimizers_G:
                optimizer_G.zero_grad()

            scaler.scale(loss_G).sum(dim=0).backward()

            for optimizer_G in optimizers_G:
                # optimizer_G.step()
                scaler.step(optimizer_G)
                scaler.update()

        for key in loss_dict.keys():
            hists_dict[key][epoch] = np.mean(loss_dict[key])

        improved = [False] * 3
        for i in range(N):

            hists_dict[val_loss_keys[i]][epoch], hists_dict[acc_keys[i]][epoch] = validate_with_label(generators[i],
                                                                                                      val_xes[i], val_y,
                                                                                                      val_labels[i])

            if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
                best_mse[i] = hists_dict[val_loss_keys[i]][epoch]
                best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                best_epoch_mse[i] = epoch + 1
                improved[i] = True
            if hists_dict[acc_keys[i]][epoch] > best_acc[i]:
                best_acc[i] = hists_dict[acc_keys[i]][epoch]
                best_epoch_acc[i] = epoch + 1

            schedulers[i].step(hists_dict[val_loss_keys[i]][epoch])

        if distill_epochs > 0 and (epoch + 1) % 30 == 0:
            # if distill and patience_counter > 1:
            losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            rank = np.argsort(losses)
            print(f"Do distill {distill_epochs} epoch! Distill from G{rank[0] + 1} to G{rank[-1] + 1}")
            logging.info(f"Do distill {distill_epochs} epoch! Distill from G{rank[0] + 1} to G{rank[-1] + 1}")
            for e in range(distill_epochs):
                do_distill(rank, generators, dataloaders, optimizers_G, window_sizes, device)

        if (epoch + 1) % 10 == 0 and cross_finetune_epochs > 0:
            G_losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            D_losses = [np.mean(loss_dict[d_keys[i]]) for i in range(N)]
            G_rank = np.argsort(G_losses)
            D_rank = np.argsort(D_losses)
            print(f"Start cross finetune!  So far G{G_rank[0] + 1} with D{D_rank[0] + 1}")
            logging.info(f"Start cross finetune!  So far G{G_rank[0] + 1} with D{D_rank[0] + 1}")
            # if patience_counter > 1:
            for e in range(cross_finetune_epochs):
                for batch_idx, (x_last, y_last, label_last) in enumerate(dataloaders[-1]):
                    x_last = x_last.to(device)
                    y_last = y_last.to(device)
                    label_last = label_last.to(device)
                    label_last = label_last.unsqueeze(-1)
                    # print(x_last.shape, y_last.shape, label_last.shape)

                    X = []
                    Y = []
                    LABELS = []

                    for gap in gaps:
                        X.append(x_last[:, gap:, :])
                        Y.append(y_last[:, gap:, :])
                        LABELS.append(label_last[:, gap:, :].long())
                    X.append(x_last.to(device))
                    Y.append(y_last.to(device))
                    LABELS.append(label_last.to(device).long())
                    cross_best_Gloss = np.inf

                    generators[G_rank[0]].eval()
                    discriminators[D_rank[0]].train()

                    loss_D, lossD_G = discriminate_fake(args, [X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[D_rank[0]]],
                                                        [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                        [window_sizes[D_rank[0]]], target_num,
                                                        criterion, weight_matrix[D_rank[0], G_rank[0]],
                                                        device, mode="train_D")

                    optimizers_D[D_rank[0]].zero_grad()

                    # loss_D.sum(dim=0).backward()
                    scaler.scale(loss_D.sum(dim=0)).backward()
                    # optimizers_D[D_rank[0]].step()
                    scaler.step(optimizers_D[D_rank[0]])
                    scaler.update()

                    discriminators[D_rank[0]].eval()
                    generators[G_rank[0]].train()

                    '''训练生成器'''
                    weight = weight_matrix[:, :-1].clone().detach()  # [N, N]
                    loss_G, loss_mse_G = discriminate_fake(args, [X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[D_rank[0]]],
                                                           [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                           [window_sizes[D_rank[0]]], target_num,
                                                           criterion, weight[D_rank[0], G_rank[0]],
                                                           device,
                                                           mode="train_G")

                    optimizers_G[G_rank[0]].zero_grad()
                    # loss_G.sum(dim=0).backward()
                    scaler.scale(loss_G.sum(dim=0)).backward()
                    # optimizers_G[G_rank[0]].step()
                    scaler.step(optimizers_G[G_rank[0]])
                    scaler.update()

                validate_G_loss, validate_G_acc = validate_with_label(generators[G_rank[0]], val_xes[G_rank[0]], val_y,
                                                                      val_labels[G_rank[0]])

                if validate_G_loss >= cross_best_Gloss:
                    generators[G_rank[0]].load_state_dict(best_model_state[G_rank[0]])
                    break
                elif validate_G_loss < cross_best_Gloss:
                    cross_best_Gloss = validate_G_loss
                    best_mse[G_rank[0]] = cross_best_Gloss
                    best_model_state[G_rank[0]] = copy.deepcopy(generators[G_rank[0]].state_dict())
                    best_epoch_mse[G_rank[0]] = epoch + 1

                print(
                    f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1} with D{D_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}, Validation Acc {validate_G_acc * 100:.2f}%")
                logging.info(
                    f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1} with D{D_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}, Validation Acc {validate_G_acc * 100:.2f}%")  # NEW

        # 动态生成打印字符串
        log_str_mse = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]:.8f}"
            for i, key in enumerate(val_loss_keys)
        )
        log_str_acc = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch] * 100:.2f} %"
            for i, key in enumerate(acc_keys)
        )
        # if len(acc_keys) == 1:
        #     best_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])
        # else:
        #     best_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])

        logging.info("Epoch %d | Validation MSE: %s | Accuracy: %s", epoch + 1, log_str_mse, log_str_acc)  # NEW
        # print(f"Patience Counter:{patience_counter}, Best MSE Epochs | {best_info}")
        print(f"Patience Counter:{patience_counter}/{patience}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        epo_end = time.time()
        print(f"Epoch time: {epo_end - epo_start:.4f}")

    data_G = [[[] for _ in range(4)] for _ in range(N)]
    data_D = [[[] for _ in range(4)] for _ in range(N)]

    # for i in range(N):
    #     for j in range(N + 1):
    #         if j < N:
    #             data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch]
    #             data_D[i][j] = hists_dict[f"D{i + 1}_G{j + 1}"][:epoch]
    #         elif j == N:
    #             data_G[i][j] = hists_dict[g_keys[i]][:epoch]
    #             data_D[i][j] = hists_dict[d_keys[i]][:epoch]
    for i in range(N):
        for j in range(N + 1):
            if j < N:
                # 修正：使用切片 [:epoch + 1] 以包含当前 epoch 的数据
                data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch + 1]
                data_D[i][j] = hists_dict[f"D{i + 1}_G{j + 1}"][:epoch + 1]
            elif j == N:
                data_G[i][j] = hists_dict[g_keys[i]][:epoch + 1]
                data_D[i][j] = hists_dict[d_keys[i]][:epoch + 1]

    plot_generator_losses(data_G, output_dir)
    plot_discriminator_losses(data_D, output_dir)

    # overall G&D
    visualize_overall_loss([data_G[i][N] for i in range(N)], [data_D[i][N] for i in range(N)], output_dir)

    # hist_MSE_G = [[] for _ in range(N)]
    # hist_val_loss = [[] for _ in range(N)]
    # for i in range(N):
    #     hist_MSE_G[i] = hists_dict[f"MSE_G{i + 1}"][:epoch]
    #     hist_val_loss[i] = hists_dict[f"val_G{i + 1}"][:epoch]
    hist_MSE_G = [[] for _ in range(N)]
    hist_val_loss = [[] for _ in range(N)]
    for i in range(N):
        hist_MSE_G[i] = hists_dict[f"MSE_G{i + 1}"][:epoch + 1]
        hist_val_loss[i] = hists_dict[f"val_G{i + 1}"][:epoch + 1]

    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch, output_dir)

    mse_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])
    acc_info = ", ".join([f"G{i + 1}:{best_epoch_acc[i]}" for i in range(N)])
    acc_value_info = ", ".join([f"G{i + 1}: {best_acc[i] * 100:.2f}%" for i in range(N)])
    mse_value_info = ", ".join([f"G{i + 1}: {best_mse[i]:.6f}" for i in range(N)])

    print(f"[Best MSE Epochs]     {mse_info}")
    print(f"[Best MSE Values]     {mse_value_info}")
    print(f"[Best ACC Epochs]     {acc_info}")
    print(f"[Best Accuracy Values]{acc_value_info}")

    logging.info(f"[Best MSE Epochs]     {mse_info}")
    logging.info(f"[Best MSE Values]     {mse_value_info}")
    logging.info(f"[Best ACC Epochs]     {acc_info}")
    logging.info(f"[Best Accuracy Values]{acc_value_info}")

    return best_acc, best_model_state


def discriminate_fake(args, X, Y, LABELS,
                      generators, discriminators,
                      window_sizes, target_num,
                      criterion, weight_matrix,
                      device,
                      mode):
    assert mode in ["train_D", "train_G"]

    N = len(generators)

    # discriminator output for real data
    with get_autocast_context(args.amp_dtype):
        # 自动混合精度上下文
        dis_real_outputs = [model(y, label) for (model, y, label) in zip(discriminators, Y, LABELS)]
        outputs = [generator(x) for (generator, x) in zip(generators, X)]  # cannot be omitted
        real_labels = [torch.ones_like(dis_real_output).to(device) for dis_real_output in dis_real_outputs]
        fake_data_G, fake_logits_G = zip(*outputs)
        # 假设 fake_logits_G 是一个 list，每个元素是 [batch_size, num_classes] 的 tensor
        fake_cls_G = [torch.argmax(logit, dim=1) for logit in fake_logits_G]  # shape: [batch_size]

        lossD_real = [criterion(dis_real_output, real_label) for (dis_real_output, real_label) in
                      zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        fake_data_temp_G = [fake_data.detach() for fake_data in fake_data_G]
        fake_data_temp_G = [torch.cat([label[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (label, window_size, fake_data) in zip(Y, window_sizes, fake_data_temp_G)]

        # ✅ 正确：在当前分支中完整定义和赋值
        fake_cls_detached = [fake_cls.detach() for fake_cls in fake_cls_G]
        fake_cls_temp_G = [torch.cat([label[:, :window_size, :], fake_cls.unsqueeze(-1).unsqueeze(1)], axis=1)
                           for (label, window_size, fake_cls) in zip(LABELS, window_sizes, fake_cls_detached)]

    elif mode == "train_G":
        fake_data_temp_G = [torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], axis=1)
                            for (y, window_size, fake_data) in zip(Y, window_sizes, fake_data_G)]

        # ✅ 正确：在当前分支中完整定义和赋值
        fake_cls_temp_G = [torch.cat([label[:, :window_size, :], fake_cls.unsqueeze(-1).unsqueeze(1)], axis=1)
                           for (label, window_size, fake_cls) in zip(LABELS, window_sizes, fake_cls_G)]

    # 判别器对伪造数据损失
    # 三个生成器的结果的数据对齐
    fake_data_GtoD = {}
    fake_cls_GtoD = {}
    for i in range(N):
        for j in range(N):
            if i < j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :window_sizes[j] - window_sizes[i], :], fake_data_temp_G[i]], axis=1)
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [LABELS[j][:, :window_sizes[j] - window_sizes[i], :], fake_cls_temp_G[i]], axis=1)
            elif i > j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_temp_G[i][:, window_sizes[i] - window_sizes[j]:, :]
            elif i == j:
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_temp_G[i]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_temp_G[i]

    fake_labels = [torch.zeros_like(real_label).to(device) for real_label in real_labels]

    with get_autocast_context(args.amp_dtype):
        # 自动混合精度上下文
        dis_fake_outputD = []
        for i in range(N):
            row = []
            for j in range(N):
                out = discriminators[i](fake_data_GtoD[f"G{j + 1}ToD{i + 1}"],
                                        fake_cls_GtoD[f"G{j + 1}ToD{i + 1}"].long())
                row.append(out)
            if mode == "train_D":
                row.append(lossD_real[i])
            dis_fake_outputD.append(row)  # dis_fake_outputD[i][j] = Di(Gj)

        if mode == "train_D":
            loss_matrix = torch.zeros(N, N + 1, device=device)  # device 取决于你的模型位置
            weight = weight_matrix.clone().detach()  # [N, N+1]
            for i in range(N):
                for j in range(N + 1):
                    if j < N:
                        loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], fake_labels[i])
                    elif j == N:
                        loss_matrix[i, j] = dis_fake_outputD[i][j]
        elif mode == "train_G":
            loss_matrix = torch.zeros(N, N, device=device)  # device 取决于你的模型位置
            weight = weight_matrix.clone().detach()  # [N, N]
            for i in range(N):
                for j in range(N):
                    loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], real_labels[i])

        loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)  # [N, N] --> [N, ]

        if mode == "train_G":
            loss_mse_G = [F.mse_loss(fake_data.squeeze(), y[:, -1, :].squeeze()) for (fake_data, y) in
                          zip(fake_data_G, Y)]
            loss_matrix = loss_mse_G
            loss_DorG = loss_DorG + torch.stack(loss_matrix).to(device)

    return loss_DorG, loss_matrix


def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device,
               *,
               alpha: float = 0.3,  # 软目标权重
               temperature: float = 2.0,  # 温度系数
               grad_clip: float = 1.0,  # 梯度裁剪上限 (L2‑norm)
               mse_lambda: float = 0.8,
               ):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    # term of teacher is longer
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y, label) in enumerate(distill_dataloader):

        y = y[:, -1, :]
        y = y.to(device)
        label = label[:, -1]
        label = label.to(device)
        if gap > 0:
            x_teacher = x
            x_student = x[:, gap:, :]
        else:
            x_teacher = x[:, (-1) * gap:, :]
            x_student = x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output, teacher_cls = teacher_generator(x_teacher)
        teacher_output, teacher_cls = teacher_output.detach(), teacher_cls.detach()
        # Forward pass with student generator
        student_output, student_cls = student_generator(x_student)

        # 使用温度缩放后计算 softmax 分布
        teacher_soft = F.softmax(teacher_cls.detach() / temperature, dim=1)
        student_log_soft = F.log_softmax(student_cls / temperature, dim=1)

        # 软标签学习损失：KL 散度
        soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (alpha * temperature ** 2)

        label_onehot = F.one_hot(label.long(), num_classes=student_cls.size(1)).float()

        # 硬目标损失：学生分类输出和真实标签计算交叉熵
        hard_loss = nn.BCEWithLogitsLoss()(student_cls, label_onehot) * (1 - alpha)
        hard_loss += F.mse_loss(student_output * temperature, y) * (1 - alpha) * mse_lambda
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        # distillation_loss.backward()
        scaler.scale(distillation_loss).backward()

        if grad_clip is not None:
            clip_grad_norm_(student_generator.parameters(), grad_clip)

        # student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed
        scaler.step(student_optimizer)
        scaler.update()


if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    # 将当前文件所在目录的上级加入 sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
