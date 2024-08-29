import pandas as pd

# ensures correct input
def inputFormatter(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt))
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value in the requested range.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Get's user input
def get_user_input():
    print("Please input the following values (all must be numbers):")
    inputs = []
    inputs.append(inputFormatter("Enter your age: ", 0))
    inputs.append(inputFormatter("Enter your sex? (female (0) or male (1)): ", 0, 1))
    inputs.append(inputFormatter("Rate chest pain (0-3): ", 0, 3)) 
    inputs.append(inputFormatter("Enter resting systolic blood pressure: "))
    inputs.append(inputFormatter("Enter serum cholesterol: "))
    inputs.append(inputFormatter("Enter fasting blood sugar (less than 126 mg/dL (0) or 126 mg/dL or greater (1)): ", 0, 1))
    inputs.append(inputFormatter("Enter your resting electrocardiographic results (0-2 on an abnormality scale): ", 0, 2)) 
    inputs.append(inputFormatter("Enter maximum heart rate achieved: "))
    inputs.append(inputFormatter("Do you have angina? (no (0) or yes (1)): ", 0, 1))
    inputs.append(inputFormatter("What is your oldpeak value?: ")) 
    inputs.append(inputFormatter("Enter the slope of your peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping): ", 0, 2))
    inputs.append(inputFormatter("Enter the number of major vessels (0-3) with significant narrowing or blockage: ", 0, 3)) 
    inputs.append(inputFormatter("Enter your thalassemia result (0 = Normal, 1 = Fixed defect, 2 = Reversible defect, 3 = Other): ", 0, 3)) 

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    user_input_df = pd.DataFrame([inputs], columns=columns)
    return user_input_df
