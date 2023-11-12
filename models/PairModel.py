import torch
import os
from torch.nn import BCEWithLogitsLoss
from models.GraphSAGE_LSTM import GL_Pair, MLPModel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.analysis import analysis

HIDDEN_CHANNELS = 128
NUM_FEATURES = 11
POSITIVE_RATE = 0.0005 #hyperparameter
NUM_CLASSES = 2
LR = 1e-3
DROPOUT = 0.3

class PairModel(torch.nn.Module):
    def __init__(self,
             train_data = None,
             valid_data = None,
             device = None,
             model_path = None
             ):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.device = device
        self.Model_Path = model_path
        weights = [(1-POSITIVE_RATE)/POSITIVE_RATE]
        class_weights = torch.FloatTensor(weights).to(device)
        self.criterion = BCEWithLogitsLoss(pos_weight = class_weights)
        self.model = GL_Pair(in_channels=NUM_FEATURES,hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT).to(device)
        self.premlp = MLPModel(in_size=HIDDEN_CHANNELS*2, hidden_size=HIDDEN_CHANNELS*2, out_size=1).to(device)
        self.optimizer = torch.optim.Adam([
        dict(params=self.model.parameters(), weight_decay=1e-2),#weight_decay=0.01
        dict(params=self.premlp.parameters(), weight_decay=5e-4)#weight_decay=5e-4 1e-2
        ], lr=LR)#lr=0.01


    def train(self, num_epochs):
        best_epoch = 0
        best_val_auc = 0
        best_val_aupr = 0
        self.model.train()
        self.premlp.train()
        for epoch in tqdm(range(num_epochs)):
            print("\n========== Train epoch " + str(epoch + 1) + " ==========")
            epoch_loss_train_avg = self.train_one_epoch()
            print("========== Evaluate Train set ==========")
            _, train_true, train_pred = self.evaluate(self.train_data)
            result_train = analysis(train_true, train_pred, 0.5)
            print("Train loss: ", epoch_loss_train_avg)
            print("Train binary acc: ", result_train['binary_acc'])
            print("Train AUC: ", result_train['AUC'])
            print("Train AUPRC: ", result_train['AUPRC'])
            print("========== Evaluate Valid set ==========")
            epoch_loss_valid_avg, valid_true, valid_pred = self.evaluate(self.valid_data)
            result_valid = analysis(valid_true, valid_pred, 0.5)
            print("Valid loss: ", epoch_loss_valid_avg)
            print("Valid binary acc: ", result_valid['binary_acc'])
            print("Valid precision: ", result_valid['precision'])
            print("Valid recall: ", result_valid['recall'])
            print("Valid f1: ", result_valid['f1'])
            print("Valid AUC: ", result_valid['AUC'])
            print("Valid AUPRC: ", result_valid['AUPRC'])
            print("Valid mcc: ", result_valid['mcc'])
            if best_val_auc < result_valid['AUC']:#AUPRC
                best_epoch = epoch + 1
                best_val_auc = result_valid['AUC']
                best_val_aupr = result_valid['AUPRC']
                torch.save({"model": self.model.state_dict(),
                   "premlp": self.premlp.state_dict()}, os.path.join(self.Model_Path, 'fullmodel.pkl'))
        return best_epoch, best_val_auc, best_val_aupr

    def train_full_model(self, aver_epoch, num_epochs):
        print("\n\nTraining a full model using all training data...\n")
        self.model.train()
        self.premlp.train()
        loss_list = []
        auc_list = []
        auprc_list = []
        for epoch in range(num_epochs):
            print("\n========== Train epoch " + str(epoch + 1) + " ==========")
            epoch_loss_train_avg = self.train_one_epoch()
            print("========== Evaluate Train set ==========")
            _, train_true, train_pred = self.evaluate(self.train_data)
            result_train = analysis(train_true, train_pred, 0.5)
            print("Train loss: ", epoch_loss_train_avg)
            print("Train AUC: ", result_train['AUC'])
            print("Train AUPRC: ", result_train['AUPRC'])
            loss_list.append(epoch_loss_train_avg)
            auc_list.append(result_train['AUC'])
            auprc_list.append(result_train['AUPRC'])
            if epoch + 1 in [aver_epoch]:
                torch.save({"model": self.model.state_dict(), "premlp": self.premlp.state_dict()}, os.path.join(self.Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))

    def train_one_epoch(self):
        epoch_loss_train = 0.0
        n = 0
        scaler = StandardScaler()
        for _, data in enumerate(self.train_data):
            h_a = None
            c_a = None
            h_b = None
            c_b = None
            data_a, data_b = data
            for time, snapshot_a in enumerate(data_a):
                snapshot_b=data_b[time]
                snapshot_a.x = torch.from_numpy(scaler.fit_transform(snapshot_a.x))
                snapshot_a.x = snapshot_a.x.to(torch.float32)
                snapshot_a = snapshot_a.to(self.device)
                snapshot_b.x = torch.from_numpy(scaler.fit_transform(snapshot_b.x))
                snapshot_b.x = snapshot_b.x.to(torch.float32)
                snapshot_b = snapshot_b.to(self.device)
                if time==15:
                    y = snapshot_a.y
                    pred_a, c_a = self.model(x=snapshot_a.x, edge_index=snapshot_a.edge_index, h=h_a,c=c_a, output= True)
                    pred_b, c_b = self.model(x=snapshot_b.x, edge_index=snapshot_b.edge_index, h=h_b, c=c_b, output=True)
                else:
                    h_a, c_a = self.model(x=snapshot_a.x, edge_index=snapshot_a.edge_index, h=h_a,c=c_a)
                    h_b, c_b = self.model(x=snapshot_b.x, edge_index=snapshot_b.edge_index, h=h_b, c=c_b)
            self.optimizer.zero_grad()
            num_nodes = len(snapshot_a.x)
            y_pred = torch.zeros(num_nodes,num_nodes).to(self.device)
            y_true = torch.zeros(num_nodes,num_nodes).to(self.device)
            for m in range(0,num_nodes):
                if y[m] != 0:
                    y_true[m][y[m]] = 1
                for j in range(0,num_nodes):
                    pred = torch.cat((pred_a[m],pred_b[j]),dim=0)
                    y_pred[m][j] = self.premlp(pred)
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
            loss = self.criterion(y_pred, y_true) 
            # backward gradient
            loss.backward()
            self.optimizer.step()
            epoch_loss_train += loss.item()
            n += 1
        epoch_loss_train_avg = epoch_loss_train / n
        return epoch_loss_train_avg
        
    def evaluate(self, data_set):
        self.model.eval()
        self.premlp.eval()
        epoch_loss = 0.0
        n = 0
        valid_pred = []
        valid_true = []
        scaler = StandardScaler()
        for _, data in enumerate(data_set):
            with torch.no_grad():
                h_a = None
                c_a = None
                h_b = None
                c_b = None
                data_a, data_b = data
                for time, snapshot_a in enumerate(data_a):
                    snapshot_b=data_b[time]
                    snapshot_a.x = torch.from_numpy(scaler.fit_transform(snapshot_a.x))
                    snapshot_a.x = snapshot_a.x.to(torch.float32)
                    snapshot_a = snapshot_a.to(self.device)
                    snapshot_b.x = torch.from_numpy(scaler.fit_transform(snapshot_b.x))
                    snapshot_b.x = snapshot_b.x.to(torch.float32)
                    snapshot_b = snapshot_b.to(self.device)
                    if time == 15:
                        y = snapshot_a.y
                        pred_a, c_a = self.model(x=snapshot_a.x, edge_index=snapshot_a.edge_index, h=h_a,c=c_a, output= True)
                        pred_b, c_b = self.model(x=snapshot_b.x, edge_index=snapshot_b.edge_index, h=h_b, c=c_b, output=True)
                    else:
                        h_a, c_a = self.model(x=snapshot_a.x, edge_index=snapshot_a.edge_index, h=h_a,c=c_a)
                        h_b, c_b = self.model(x=snapshot_b.x, edge_index=snapshot_b.edge_index, h=h_b, c=c_b)
                self.optimizer.zero_grad()
                num_nodes = len(snapshot_a.x)
                y_pred = torch.zeros(num_nodes,num_nodes).to(self.device)
                y_true = torch.zeros(num_nodes,num_nodes).to(self.device)
                for m in range(0,num_nodes):
                    if y[m] != 0:
                        y_true[m][y[m]] = 1
                    for j in range(0,num_nodes):
                        pred = torch.cat((pred_a[m],pred_b[j]),dim=0)
                        y_pred[m][j] = self.premlp(pred)
                y_pred = y_pred.view(-1)
                y_true = y_true.view(-1)
                loss = self.criterion(y_pred, y_true) #mask
                sigmoid = torch.nn.Sigmoid()
                y_pred = sigmoid(y_pred)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = y_true.cpu().detach().numpy()
                valid_pred += [pr for pr in y_pred]
                valid_true += list(y_true)
                epoch_loss += loss.item()
                n += 1
        epoch_loss_avg = epoch_loss / n

        return epoch_loss_avg, valid_true, valid_pred #pred_dict
    
    def test(self, test_set):
        for model_name in sorted(os.listdir(self.Model_Path)):
            print(model_name)
            checkpoint = torch.load(self.Model_Path + model_name, map_location= self.device)#'cuda:0'
            self.model.load_state_dict(checkpoint["model"])
            self.premlp.load_state_dict(checkpoint["premlp"])
            epoch_loss_test_avg, test_true, test_pred = self.evaluate(test_set)
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
