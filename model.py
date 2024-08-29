import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

epochs = []

# Cuda or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data from the Cleveland Clinic Foundation
data = pd.read_csv("heart.csv")

x = data.drop(columns=["target"])
y = data["target"]

# split to test and train
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

# standardize data
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# convert data
xTrain = torch.tensor(xTrain, dtype=torch.float32).to(device)
xTest = torch.tensor(xTest, dtype=torch.float32).to(device)
yTrain = torch.tensor(yTrain.values, dtype=torch.float32).to(device)
yTest = torch.tensor(yTest.values, dtype=torch.float32).to(device)


# architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


input_size = xTrain.shape[1]
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 100

# initialize neural network
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train neural network
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(xTrain)
    loss = criterion(outputs, yTrain.view(-1, 1))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predicted = outputs.round()
        correct = (predicted == yTrain.view(-1, 1)).float().sum()
        accuracy = correct / yTrain.size(0)

    if (epoch + 1) % 10 == 0:
        epochs.append(f"Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():4f}, Accuracy: {accuracy.item() * 100:.2f}%")


torch.save(model, "heart_disease_model.pth")
joblib.dump(scaler, "scaler.pkl")

def getAccuracy():
     
    for i in  epochs:
        print(i)

    # model evaluation
    # 86.89% accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(xTest)
        predicted = outputs.round()
        correct = (predicted == yTest.view(-1, 1)).float().sum()
        accuracy = correct / yTest.size(0)
        print(f"Accuracy of Model: {accuracy.item() * 100:.2f}%")