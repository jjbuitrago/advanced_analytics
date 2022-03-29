import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime


from sklearn.model_selection import train_test_split

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Define a class to create the DataLoader
class MyDataset(torch.utils.data.Dataset):

  def __init__(self,df_x, df_y):
    self.x_train=torch.tensor(df_x,dtype=torch.float32)
    self.y_train=torch.tensor(df_y,dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx] 

## Process the csv file
train_df = pd.read_csv("data/train_month_3_with_target.csv")

train_df = train_df.dropna(axis = 1)

y=train_df["target"]

# The data cleaning is not too good
X = train_df.drop(["target", "client_id"], axis = 1)
# X = X.drop(["customer_since_all","customer_since_bank","customer_birth_date", "customer_children","customer_relationship"], axis = 1) # For now
for col in ["customer_since_all","customer_since_bank","customer_birth_date", "customer_children","customer_relationship"]:
    try:
        X = X.drop(col, axis = 1)
    except:
        pass

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.33, random_state=42)

training_set=MyDataset(X_train.values, y_train.values)
validation_set=MyDataset(X_val.values, y_val.values)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=5, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=5, shuffle=False, num_workers=2)

# Hyper parameters
EPOCHS = 8
learning_rate = 0.01

# My model
class Net(torch.nn.Module):
    def __init__(self, x_train_shape):
        super(Net, self).__init__()
        # An affine operation: y = Wx + b de tipo todos contra todos
        self.Layer_1 = nn.Linear(x_train_shape, 20) 
        self.Layer_2 = nn.Linear(20, 5)
        self.Layer_Output = nn.Linear(5, 1)  
        
        # Define sigmoid activation and softmax output 
        self.Tanh = nn.Tanh()
        
    def forward(self, inputs):
        inputs = self.Layer_1(inputs)
        inputs = self.Tanh(inputs)
        inputs = self.Layer_2(inputs)
        return self.Layer_Output(inputs)

# Define the model, loss criterion and optimizer
model = Net(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Function to train one epoch
def train_one_epoch(epoch_index): #, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        try:
            loss = criterion(outputs.view(5), labels)
        except RuntimeError:
            continue
        
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0
best_vloss = 1_000_000.

# Let the training begin
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number) #, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    # writer.add_scalars('Training vs. Validation Loss',
    #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
    #                 epoch_number + 1)
    # writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        print(model_path)

    epoch_number += 1

# Use the test set
test_df = pd.read_csv("data/test_month_3.csv")

df_pred = pd.DataFrame()
df_pred["ID"] = test_df["client_id"]

test_df = test_df.dropna(axis = 1)

X_test = test_df.drop(["client_id"], axis = 1)
# X = X.drop(["customer_since_all","customer_since_bank","customer_birth_date", "customer_children","customer_relationship"], axis = 1) # For now
for col in ["customer_since_all","customer_since_bank","customer_birth_date", "customer_children","customer_relationship"]:
    try:
        X_test = X_test.drop(col, axis = 1)
    except:
        pass

# Make a prediction
new_data = torch.tensor(X_test.values).type(torch.FloatTensor)
with torch.no_grad():
    prediction = model(new_data)

df_pred["PROB"] = prediction.view(27300).data.detach().numpy()

df_pred.to_csv(f"{timestamp}_attempt.csv", index = None)