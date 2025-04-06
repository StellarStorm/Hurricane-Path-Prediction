import geopy.distance
import torch
from matplotlib import pyplot as plt
from numpy.random import shuffle
from torch.nn import MSELoss
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm

from utils.coords_parser import coords_parser
from utils.preprocessing import remove_small_samples, train_val_test_split
from utils.torch_rnn import HurricaneRNN


def test_pacific_lon(size, batch_size=16, device: str = 'cpu'):
    model = HurricaneRNN('Atlantic', size - 1, 0.2).to(device)

    model.load_state_dict(
        torch.load(
            f'models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_{device}_short_term_5pts.h5'
        )
    )
    model.eval()
    #model.load_state_dict(torch.load("models/HurricaneRes_RNN_2D_Atlantic_1LSTMS_256_cpu_short_term_5pts.h5"))

    dict_atlantique = coords_parser('Data/atlantic_latest.csv')
    dict_atlantique = remove_small_samples(dict_atlantique, min_size=size)

    train_keys, val_keys, test_keys = train_val_test_split(list(dict_atlantique.keys()))
    x_test = []
    y_test = []
    for key in test_keys:
        for i in range(len(dict_atlantique[key])-size+1):
            x_test.append(dict_atlantique[key][i:size-1+i])
            y_test.append(dict_atlantique[key][size - 1 + i:size + i])
    nb_data_test = len(x_test)
    permut = list(range(nb_data_test))
    shuffle(permut)
    x_test_, y_test_ = [x_test[i] for i in permut], [y_test[i] for i in permut]
    x_test, y_test = x_test_, y_test_
    if not nb_data_test == len(y_test):
        raise ValueError(f" X and Y must be of same length. Found x : {nb_data_test} and y : {len(y_test)}")

    x_test = torch.stack(x_test, dim=0).to(device).float()
    y_test = torch.stack(y_test, dim=0).to(device).float()

    nb_iters_test = nb_data_test // batch_size
    mean_rmse = 0.0
    mean_mae= 0.0
    mean_dist= 0.0
    rmse = MSELoss().to(device)
    mae = MeanAbsoluteError().to(device)

    result = []
    with torch.inference_mode():
        for i in tqdm(range(nb_iters_test)):
            data = x_test[i * batch_size : (i + 1) * batch_size, ...]
            gt = y_test[i * batch_size : (i + 1) * batch_size, ...]
            out = model(data)
            out = torch.add(
                torch.multiply(out, torch.tensor([[10, 20]], device=device)),
                torch.tensor([[27, -65]], device=device),
            )
            gt = torch.add(
                torch.multiply(gt, torch.tensor([[[10, 20]]], device=device)),
                torch.tensor([[[27, -65]]], device=device),
            )
            rmse_value = torch.sqrt(rmse(torch.unsqueeze(out, dim=1), gt))
            mae_value = mae(torch.unsqueeze(out, dim=1), gt)
            mean_rmse += rmse_value.item()
            mean_mae += mae_value.item()
            lat_dist = torch.mean(59.9 * (out[:, 0] - gt[:, :, 0]))
            lon_dist = torch.mean(47.79 * (out[:, 0] - gt[:, :, 0]))
            for i_ in range(batch_size):
                lat1 = out[i_, 0].detach()
                lon1 = out[i_, 1].detach()
                lat2 = gt[i_, 0, 0].detach()
                lon2 = gt[i_, 0, 1].detach()
                mean_dist += geopy.distance.geodesic(
                    (lat1, lon1), (lat2, lon2)
                ).nautical
                # print(mean_dist)
            result.append(
                (
                    x_test[i * batch_size : (i + 1) * batch_size],
                    gt,
                    out.detach(),
                )
            )
    print(f"Test RMSE {mean_rmse / nb_iters_test}")
    print(f"Test MSE {(mean_rmse / nb_iters_test)**2}")
    print(f"Test MAE {mean_mae / nb_iters_test}")
    print(f"Test Distance (approx.):  {mean_dist / (nb_iters_test * batch_size)} n miles")
    return result

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    points = test_pacific_lon(size=5, batch_size=16)

    for batch in points:
        x, y, y_hat = batch
        for i in range(len(x)):
            x[i] = torch.add(
                torch.multiply(x[i], torch.tensor([[10, 20]])),
                torch.tensor([[27, -65]]),
            )
            for vec in x[i].detach().cpu().numpy():
                plt.scatter(vec[1] , vec[0] * 1, c="blue")
            plt.scatter(
                y[i].detach().cpu().numpy()[0][1],
                y[i].detach().cpu().numpy()[0][0],
                c='red',
            )
            plt.scatter(
                y_hat[i].detach().cpu().numpy()[1],
                y_hat[i].detach().cpu().numpy()[0],
                c='pink',
            )
        plt.show()
