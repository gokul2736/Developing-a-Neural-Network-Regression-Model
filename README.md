# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

### Name: Markandeyan Gokul

### Register Number: 212224240086


## PROGRAM
```python

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Exp-1.csv")

X = df[["Input"]].values
y = df[["Output"]].values

# Normalize data
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Create neural network
model = nn.Sequential(
    nn.Linear(1, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 1000
losses = []

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Plot training loss vs iteration
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Iteration (Epoch)")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_regression.png")
plt.show()

# Predict outputs
model.eval()

with torch.no_grad():
    predictions = model(X).numpy()

predictions = y_scaler.inverse_transform(predictions)

print("\nActual vs Predicted Output")
for actual, predicted in zip(df["Output"], predictions.flatten()):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

print("\nRegression experiment completed successfully!")

```
**Output**
  <img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/bea880a9-7d87-4519-a4c5-a46a2c40a327" />
     


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
