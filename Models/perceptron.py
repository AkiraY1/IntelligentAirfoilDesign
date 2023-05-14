from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import torch

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None, sep='\t')
        df1 = df.sample(frac = 0.8)
        df2 = df.drop(df1.index)
        # store the inputs and outputs
        self.X = df1.values[:, -3:].astype('float32')
        self.y = df1.values[:, :-3].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 5))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
 
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 8)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(8, 16)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer
        self.hidden3 = Linear(16, 32)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        #Fourth hidden layer
        self.hidden4 = Linear(32, 16)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()
        #Fifth hidden layer and output
        self.hidden5 = Linear(16, 5)
        xavier_uniform_(self.hidden5.weight)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden5(X)
        return X
 
# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl
 
# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(500):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print(f"Epoch {epoch}: {loss}")
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data
path = 'snappy.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(3)
# train the model
train_model(train_dl, model)

# If running on Google Colab, also run:
# !mkdir -p saved_model

torch.save(model, "saved_model/my_model")

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 5))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    i = 0
    r1, r2, r3, r4, r5 = 0, 0, 0, 0, 0
    for pred in predictions:
        r1 += abs(pred[0] - actuals[i][0])
        r2 += abs(pred[1] - actuals[i][1])
        r3 += abs(pred[2] - actuals[i][2])
        r4 += abs(pred[3] - actuals[i][3])
        r5 += abs(pred[4] - actuals[i][4])
        i += 1
    mse = mean_squared_error(actuals, predictions)
    print(i)
    return mse, r1, r2, r3, r4, r5

mse, r1, r2, r3, r4, r5 = evaluate_model(test_dl, model)
print(f"thickness: {r1}")
print(f"thickness_pos: {r2}")
print(f"camber: {r3}")
print(f"camber_pos: {r4}")
print(f"alpha: {r5}")
