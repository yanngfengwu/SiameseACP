from configuration import config as cf

def evaluate_accuracy(data_iter, net):
    config = cf.get_train_config()
    device = config.device
    acc_sum, n = 0.0, 0
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
#         for i in range(len(z)):
#             if i == 0:
#                 vec = torch.tensor(seq2vec[z[0]]).to(device)
#             else:
#                 vec = torch.cat((vec, torch.tensor(seq2vec[z[i]]).to(device)), dim=0)
        outputs = net.trainModel(x)

        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n