import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import os

def f(k, dt_o, dt_h, x, y, R):
    dt_o = float(dt_o.iloc[0])
    dt_h = float(dt_h.iloc[0])
    x = float(x.iloc[0])
    y = float(y.iloc[0])
    R = float(R.iloc[0])
    retVal = dt_o * (((1 + R * (k ** 2)) / (1 + R)) ** x) + (k ** y) * (dt_h - dt_o)
    return retVal


from config import DATA_DIR

tfo_parameters_df = pd.read_csv(DATA_DIR / 'tfo_parameters.csv')

for filename in os.listdir(DATA_DIR / 'all_time_series'):
    if filename.endswith(".csv"):
        file_path = DATA_DIR / 'all_time_series' / filename
        equipment_df = pd.read_csv(file_path)

        equipment_name = filename.split('.csv')[0]
        print('\n' + str(equipment_name) + ': OK \n')

        TFO = tfo_parameters_df[tfo_parameters_df['equipmentId'] == int(equipment_name)]
        equipment_df['K'] = abs(equipment_df['load'] / (float(TFO['nominalLoad'].iloc[0]) * 1000))

        #For a white box model, no Train/Test split is required.
        equipment_df['Temp Pred'] = equipment_df.apply(lambda x: float(f(k=x['K'],
                                                                         dt_o=TFO['deltaTopOil'],
                                                                         dt_h=TFO['deltaHotspot'],
                                                                         x=TFO['x'],
                                                                         y=TFO['y'],
                                                                         R=TFO['copperLosses'] / TFO[
                                                                             'noLoadLosses'])) + float(
            x['temperature']), axis=1)

        equipment_df['dateTime'] = pd.to_datetime(equipment_df['dateTime'])

        file_path = filename
        equipment_df.to_csv(DATA_DIR / 'results' / f'{str(equipment_name)}.csv')
    else:
        print(str(equipment_name) + ': Not Represented in Data')

print('-------------------------------------')

# Directory for storing RMSE results
rmse_results = {}

for filename in os.listdir(DATA_DIR / 'results'):
    if filename.endswith(".csv"):
        file_path = DATA_DIR / 'results' / filename
        equipment_df = pd.read_csv(file_path)

        # Define actual and predicted temperature columns
        y_true = equipment_df['hotspotTemperature']  # Replace with actual temperature column name if different
        y_pred = equipment_df['Temp Pred']

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_results[filename] = rmse
        print(f'{filename} - RMSE: {rmse}')

# Save RMSE results to a file
with open('rmse_results.txt', 'w') as f:
    for key, value in rmse_results.items():
        f.write(f'{key}: {value}\n')

print('RMSE results saved to rmse_results.txt')
