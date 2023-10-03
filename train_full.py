import numpy as np
import os
import torch
from tqdm import *
from sklearn.preprocessing import StandardScaler
from utils.site_dataset import IDPDataset
from utils.analysis import analysis
from models.GCN_LSTM import GCN_LSTM
#import torch_geometric.transforms as T

HIDDEN_CHANNELS = 128
L_R=1e-3
DROUPOUT=0.3
NUMBER_EPOCHS = 50
NUM_FEATURES = 11
POSITIVE_RATE = 0.125 #hyperparameter
#transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
Dataset_Path = "./"
Model_Path = "./models/site_models"
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

def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    for _, data in enumerate(data_loader):
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
        model.optimizer.zero_grad()
        loss = model.criterion(y_pred, y_true)
        # backward gradient
        loss.backward()
        model.optimizer.step()
        epoch_loss_train += loss.item()
        n += 1
    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg

def train_full_model(train_set, aver_epoch):
    print("\nTraining a full model using all training data...\n")
    model = GCN_LSTM(
        in_channels=NUM_FEATURES,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROUPOUT,
        num_classes=2,
        positive_rate=POSITIVE_RATE
        ).to(device)
    
    train_loader = train_set
    loss_list = []
    auc_list = []
    auprc_list = []
    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])
        loss_list.append(epoch_loss_train_avg)
        auc_list.append(result_train['AUC'])
        auprc_list.append(result_train['AUPRC'])
        if epoch + 1 in [aver_epoch]:
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))

def main():
    test_list = [0,1,2,23,31]
    train_list = [i for i in range(35) if i not in test_list]
    dataset = IDPDataset(Dataset_Path)
    train_set=dataset.index_select(train_list)
    train_full_model(train_set,43)

if __name__ == "__main__":
    main()
