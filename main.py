import torch
import torch.optim as optim
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

from model import HGTLP
from utils import (
    fix_seed, loader, split_data, get_test_data,
    get_train_graph, links_to_subgraphs, to_hypergraphs, move_to_device
)

# from timer import Timer

hop_set = 2
seeds = range(3124, 3134)
datalist = ['enron10']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = [32, 32, 32, 1]
heads = 1
hidden_size = 128
batch = 50
test_ratio = 0
lr = 0.0001
num_epochs = 200
early_stop = 20
is_new = False
is_multi = True
dropout = True

if __name__ == '__main__':
    # os.makedirs("time_log", exist_ok=True)
    # os.makedirs("AUC_AP_log", exist_ok=True)

    for data_name in datalist:

        # time_log_path = f"./time_log/{data_name}_{hop_set}_time_log.txt"
        # score_log_path = f"./AUC_AP_log/{data_name}_{hop_set}_AUC_AP_log.txt"
        # f_time = open(time_log_path, 'a')
        # f_score = open(score_log_path, 'a')

        mean_best_auc = []
        mean_best_ap = []

        for seed in seeds:
            # print(f"\n--- Running seed {seed} on dataset {data_name} ---")
            fix_seed(seed)
            data, test_len, trainable_feat = loader(data_name)
            num_nodes = data['num_nodes']
            time_steps = data['time_length']

            train_shots = list(range(0, time_steps - test_len))
            test_shots = list(range(time_steps - test_len, time_steps))

            train_graph_list = get_train_graph(data, train_shots)
            train_pos, train_neg = split_data(data,
                                              train_shots[-1],
                                              test_ratio,
                                              is_new)
            test_pos, test_neg = get_test_data(data, test_shots, is_new, is_multi)

            train_subgraphs_list, test_subgraphs_list, max_num_node_labels = links_to_subgraphs(
                train_graph_list, train_pos, train_neg, test_pos, test_neg, hop_set)

            train_hypergraphs_list = to_hypergraphs(train_subgraphs_list, max_num_node_labels, batch, True)
            test_hypergraphs_list = to_hypergraphs(test_subgraphs_list, max_num_node_labels, batch, False)

            train_hypergraphs_list = move_to_device(train_hypergraphs_list, device)
            test_hypergraphs_list = move_to_device(test_hypergraphs_list, device)

            D = int(max_num_node_labels + 1)
            input_dim = 2 * D
            edge_input_dim = D

            model = HGTLP(
                input_dim=input_dim,
                edge_input_dim=edge_input_dim,
                hidden_size=hidden_size,
                latent_dim=latent_dim,
                num_window=len(train_shots),
                with_dropout=dropout,
                heads=heads,
            ).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=5e-4
            )

            best_auc = 0
            best_ap = 0
            stop = 0

            # f_time.write(f"\n# ============================\n")
            # f_time.write(f"# seed : {seed}\n")
            # f_time.write(f"# ============================\n")

            # timer = Timer()

            for epoch in range(num_epochs):
                # timer.start_epoch()

                total_loss = []
                all_targets = []
                all_scores = []

                model.train()
                for hypergraph_list in train_hypergraphs_list:
                    y = hypergraph_list[0].y.to(device)
                    out = model(hypergraph_list)

                    loss = F.nll_loss(out, y)
                    total_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                for hypergraph_list in test_hypergraphs_list:
                    y = hypergraph_list[0].y
                    out = model(hypergraph_list)
                    all_targets.extend(y.tolist())
                    all_scores.append(out[:, 1].cpu().detach())

                mean_loss = float(np.mean(total_loss))
                all_targets = np.array(all_targets)
                all_scores = torch.cat(all_scores).cpu().numpy()
                ap = metrics.average_precision_score(all_targets, all_scores)
                fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                # print(f"Epoch {epoch:03d} | Loss: {mean_loss:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")
                # f_time.write(f"Epoch {epoch + 1}: loss={mean_loss:.4f}  AUC={auc:.4f}  AP={ap:.4f}\n")
                # timer.end_epoch(epoch,f_time)

                if auc > best_auc:
                    best_auc = auc
                    best_ap = ap
                    stop = 0
                else:
                    stop += 1
                    if stop > early_stop:
                        break
                # print("epoch end")

            # timer.saveTotal(time_log_path, extra_info=f"{data_name} hop={hop_set} seed={seed}")
            # f_score.write(f"{best_auc:.4f}\t{best_ap:.4f}\n")
            mean_best_auc.append(best_auc)
            mean_best_ap.append(best_ap)

        # f_score.write("{:.4f}\t{:.4f}\n".format(
        #     np.mean(mean_best_auc), np.mean(mean_best_ap)))
        # f_time.write("Mean AUC/AP: {:.4f} / {:.4f}\n".format(
        #     np.mean(mean_best_auc), np.mean(mean_best_ap)))
        # f_time.close()
        # f_score.close()
