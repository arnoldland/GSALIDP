# Description: evaluate the test set
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from utils.IDPdataset import SiteDataset, PairDataset
from utils.analysis import analysis
from models.GraphSAGE_LSTM import GraphSAGE_LSTM
from models.PairModel import PairModel
import argparse 

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

def test(test_set, Model_Path):
    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        model = GraphSAGE_LSTM().to(device)
        model.load_state_dict(torch.load(Model_Path + model_name, map_location=device))

        epoch_loss_test_avg, test_true, test_pred = evaluate(model, test_set)

        result_test = analysis(test_true, test_pred)

        print("========== Evaluate on the Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])

def main(predict_type):
    test_list = [0,1,2,23,31]
    
    if predict_type == 'site':
        dataset = SiteDataset(Dataset_Path)
        test_set = dataset.index_select(test_list)
        Model_Path = "./models/site_models/"
        test(test_set, Model_Path = "./models/site_models/")

    else:
        if predict_type == 'pair':
            dataset = PairDataset(Dataset_Path)
            test_set = dataset.index_select(test_list)
            Model_Path = "./models/pair_models/"
            MODEL = PairModel(valid_data = test_set, device=device ,model_path= Model_Path)
            MODEL.test(test_set)
        
        else:
            print("Wrong prediction type!")
            exit()


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='GraphSAGE-LSTM for site prediction')
    parser.add_argument('--ptype', default='site', type=str, help='site or pair')

    args = parser.parse_args()
    main(args.ptype)