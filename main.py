%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "parkinsons.csv"
data = pd.read_csv(file_path)

features = ['MDVP:Shimmer', 'MDVP:Fo(Hz)']  # Input features
target = 'status' 

x = data[features]
y = data[target]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

import joblib

joblib.dump(model, "parkinsons_model.joblib")

