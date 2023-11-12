import numpy as np
import os
import torch
from tqdm import *
from sklearn.preprocessing import StandardScaler
from utils.IDPdataset import SiteDataset, PairDataset
from utils.analysis import analysis
from models.GraphSAGE_LSTM import GraphSAGE_LSTM
from models.PairModel import PairModel
import argparse 
from sklearn.model_selection import KFold

NUMBER_EPOCHS = 100
FOLD_NUMBER = 29
Dataset_Path = "./"


SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()

def evaluate(model, data_set):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    for _,data in enumerate(data_set):
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

def train_one_epoch(model, train_set):
    epoch_loss_train = 0.0
    n = 0
    for _, data in enumerate(train_set):
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


def train(model, train_data, valid_data, model_path, fold = 0):
    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    model.train()
    for epoch in tqdm(range(NUMBER_EPOCHS)):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        epoch_loss_train_avg = train_one_epoch(model, train_data)
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred = evaluate(model, train_data)
        result_train = analysis(train_true, train_pred, 0.5)
        print("Train loss: ", epoch_loss_train_avg)
        print("Train binary acc: ", result_train['binary_acc'])
        print("Train AUC: ", result_train['AUC'])
        print("Train AUPRC: ", result_train['AUPRC'])

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred = evaluate(model, valid_data)
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
            torch.save(model.state_dict(), os.path.join(model_path, 'Fold' + str(fold) + '_best_model.pkl'))

    return best_epoch, best_val_auc, best_val_aupr



def cross_validation(all_data, cvtype, device, model_path, fold_number=5):
    kfold = KFold(n_splits = fold_number, random_state=SEED, shuffle = True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []
    all_index=range(len(all_data))
    for train_index, valid_index in kfold.split(all_index):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_data = all_data.index_select(train_index.tolist())
        valid_data = all_data.index_select(valid_index.tolist())
        print("Train on", str(train_index), "samples, validate on", str(valid_index),
              "samples")
        if cvtype == 'site':
            model = GraphSAGE_LSTM()
            model = model.to(device)
            best_epoch, valid_auc, valid_aupr = train(model, train_data, valid_data, model_path,fold + 1)
        else:
            if cvtype == 'pair':
                model = PairModel(train_data = train_data, valid_data = valid_data, device=device ,model_path= model_path)
                best_epoch, valid_auc, valid_aupr = model.train(num_epochs= NUMBER_EPOCHS)
            else:
                print("Wrong prediction type!")
                exit()
        
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number),best_epochs,valid_aucs,valid_auprs


def main(cv_type):
    test_list = [0,1,2,23,31]
    train_list = [i for i in range(34) if i not in test_list]

    if cv_type == 'site':
        Model_Path = "./models/site_models/"
        dataset = SiteDataset(Dataset_Path)
        train_set=dataset.index_select(train_list)
        #FOLD_NUMBER = 29 for site cv
    else:
        if cv_type == 'pair':
            Model_Path = "./models/pair_models/"
            dataset = PairDataset(Dataset_Path)
            train_set = dataset.index_select(train_list)
            #FOLD_NUMBER = 10 for pair cv
        else:
            print("Wrong prediction type!")
            exit()

    aver_epoch,best_epochs,valid_aucs,valid_auprs= cross_validation(train_set, 
                                                                    cvtype = cv_type, 
                                                                    device=device,
                                                                    model_path = Model_Path,
                                                                    fold_number=FOLD_NUMBER)
    print("\n\nAverage best epoch: {:d}".format(aver_epoch))
    print("Average AUC of {} fold: {:.4f}".format(FOLD_NUMBER, sum(valid_aucs) / FOLD_NUMBER))
    print("Average AUPR of {} fold: {:.4f}".format(FOLD_NUMBER, sum(valid_auprs) / FOLD_NUMBER))

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='GraphSAGE-LSTM for cross validation')
    parser.add_argument('--cvtype', default='site', type=str, help='site or pair')

    args = parser.parse_args()
    main(args.cvtype)

    