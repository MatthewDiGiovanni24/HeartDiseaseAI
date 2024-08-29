import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import joblib
from model import getAccuracy
from input_handler import get_user_input

# Matthew DiGiovanni
# August 2024

def main():

    model = torch.load("heart_disease_model.pth")
    model.eval() 
    scaler = joblib.load("scaler.pkl")

    while True:
        print("\n1. Test Model")
        print("2. Input personal data")
        print("3. Exit")
        choice = input("Enter Choice (1-4): ")

        if choice == "1":
            getAccuracy()
            
        elif choice == "2":
            user_input = get_user_input()
            user_input_scaled = scaler.transform(user_input)
            user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

            with torch.no_grad():
                output = model(user_input_tensor)
                prediction = (output > 0.5).float()

            if prediction.item() == 1:
                print("\nPrediction: High risk of heart disease.")
            else:
                print("\nPrediction: Low risk of heart disease.")
           
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice (1-3)")

if __name__ == "__main__":
    main()
