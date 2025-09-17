import torch


# Compute Transformer Temperature & Risk Level

def white_box_model(theta_a, theta_or, theta_hr, x_param, y_param, R=0.1, K=0.05):
    #theta_a = x[:, 3]  # ambient temperature
    #theta_or = x[:, 4]  # delta top oil
    #theta_hr = x[:, -2]  # heat run test y (assumption)
    #x_param = x[:, -3]  # heat run test x (assumption)
    #y_param = x[:, -1]  # heat run test gradient (assumption)

    white_box_pred = ((1 + R * K ** 2) / (1 + R)) ** x_param * \
                     (theta_or - theta_a) + K ** y_param * (theta_hr - theta_or)
    return white_box_pred

# R = float(TFO['copperLosses'].iloc[0])/float(TFO['noLoadLosses'].iloc[0])
# equipment_df['K'] = abs(equipment_df['load'] / (float(TFO['nominalLoad'].iloc[0]) * 1000))


def predict_temperature(data):
    """
    Predict the temperature of a transformer based on input data.

    Parameters:
    data (tuple): A tuple containing the following elements:
        - load_factor (float): The load factor of the transformer.
        - ambient_temp (float): The ambient temperature.
        - oil_temp (float): The oil temperature.
        - humidity (float): The humidity level.
        - wind_speed (float): The wind speed.

    Returns:
    float: The predicted temperature of the transformer.
    """
    load_factor, ambient_temp, oil_temp, humidity, wind_speed = data
    R, x, y = 0.8, 1.2, 0.9  # Transformer Constants
    base_temp = ((1 + R * load_factor ** 2) / (1 + R)) ** x * \
                (oil_temp - ambient_temp) + load_factor ** y * (110 - oil_temp)

    X_input = torch.tensor([data], dtype=torch.float32).to(device)
    correction = model(X_input).item()
    final_temp = base_temp + correction
    return final_temp


# Determine Risk Level & Strategy

def evaluate_substation(substation):
    """
    Evaluate the risk level of a substation based on predicted temperature.

    Parameters:
    substation (str): The identifier of the substation.

    Returns:
    dict: A dictionary containing the evaluation results, including:
        - Substation (str): The identifier of the substation.
        - Predicted Temperature (float): The predicted temperature of the transformer.
        - Max Load (float): The maximum load of the substation.
        - Current Load (float): The current load of the substation.
        - Status (str): The risk status of the substation ("Safe", "Warning", or "Critical").
    """
    data, max_load = fetch_data(substation)
    if data is None:
        return None

    predicted_temp = predict_temperature(data)

    # Define Safety Thresholds
    status = "Safe"
    if predicted_temp >= 90:
        status = "Warning"
    if predicted_temp >= 105 or data[0] * max_load > max_load:
        status = "Critical"

    return {
        "Substation": substation,
        "Predicted Temperature": round(predicted_temp, 2),
        "Max Load": max_load,
        "Current Load": data[0] * max_load,
        "Status": status
    }


def physics_loss(model, x_batch, k_batch, dt_o, dt_h, x_param, y_param, R, ambient_temp):
    # Calculate physical constraint
    physical_values = white_box_model(k_batch, dt_o, dt_h, x_param, y_param, R)
    physical_values = physical_values + ambient_temp

    # Get model predictions
    pred = model(x_batch)

    # Calculate physics-informed loss
    phys_loss = torch.mean((pred - physical_values) ** 2)

    return phys_loss
