# Description: evaluate the test set
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from utils.site_dataset import IDPDataset
from utils.analysis import analysis
from models.GCN_LSTM import GCN_LSTM

Dataset_Path = "./"
Model_Path = "./models/site_models/"

HIDDEN_CHANNELS = 128
NUM_FEATURES = 11
POSITIVE_RATE = 0.125 #hyperparameter
SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    for _,data in enumerate(data_loader):
        with torch.no_grad():
            h = None
            c = None
            for time, snapshot in enumerate(data):
                snapshot.x = torch.from_numpy(scaler.fit_transform(snapshot.x))
                snapshot.x = snapshot.x.to(torch.float32)
                snapshot = snapshot.to(device)
                if time==15: #output in the last time step
                    y_true = snapshot.y
                    y_pred,c = model(x=snapshot.x, 
                                    edge_index=snapshot.edge_index,
                                    h=h,
                                    c=c, 
                                    output= True)
                else:
                    h,c = model(x=snapshot.x,
                                edge_index=snapshot.edge_index,
                                h=h,
                                c=c,
                                output= False)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            epoch_loss += loss.item()
            n += 1
        epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred

def test(test_set):
    test_loader = test_set
    #DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = GCN_LSTM(
            in_channels=NUM_FEATURES,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=2,
            positive_rate=POSITIVE_RATE
            ).to(device)
        model.load_state_dict(torch.load(Model_Path + model_name, map_location=device))

        epoch_loss_test_avg, test_true, test_pred = evaluate(model, test_loader)

        result_test = analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])

def main():
    dataset = IDPDataset(Dataset_Path)
    test_list = [0,1,2,23,31]
    test_set = dataset.index_select(test_list)
    test(test_set)

if __name__ == "__main__":
    main()
