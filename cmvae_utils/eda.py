"""
file: eda.py
author: Kirsten Richardson
date: 2021

Use this script to inspect data distribution periodically whilst collecting data with manual driving using teleop.
Can adjust manual driving based on current distribution e.g. compensate for skew
"""

import pandas as pd
import matplotlib.pyplot as plt

data_path = '/home/campus.ncl.ac.uk/b3024896/Projects/gym-donkeytrack/logs/vae_test_data/state_data.csv'

# read in csv data
df = pd.read_csv(data_path, header=None)

# look at first few rows
print(df.head(5))

# check for any NaNs in dataframe 
nan_rows = pd.isnull(df).any(1).to_numpy().nonzero()[0]
if nan_rows.size == 0:
    print('There are no rows with NaN values')
else:
    print('Index of rows with NaNs:')
    for row in nan_rows:
        print(row + 1)

# pull out columns
r_col = df.iloc[:, 0]
theta_col = df.iloc[:, 1]
psi_col = df.iloc[:, 2]

# get minimum and maximum values
max_r = r_col.max()
min_r = r_col.min()

max_theta = theta_col.max()
min_theta = theta_col.min()

max_psi = psi_col.max()
min_psi = psi_col.min()

print('R: min = {}, max = {}, Theta: min = {}, max = {}, Psi: min = {}, max = {}'.format(min_r, max_r, min_theta, max_theta, min_psi, max_psi))

# see spread of values with histogram
r_col.plot.hist()
plt.show()

theta_col.plot.hist()
plt.show()

psi_col.plot.hist()
plt.show()

# once decided on ranges want to use based on min, max and spread, check which rows are outside of range (edit as needed) 
print('Index of rows with R < 0: {}'.format(df[df.iloc[:, 0] < 0.1].index + 1))

print('Index of rows with R > 30: {}'.format(df[df.iloc[:, 0] > 30].index + 1))

print('Index of rows with Theta < -90: {}'.format(df[df.iloc[:, 1] < -90].index + 1))

print('Index of rows with Theta > 90: {}'.format(df[df.iloc[:, 1] > 90].index + 1))

print('Index of rows with Psi < -90: {}'.format(df[df.iloc[:, 2] < -90].index + 1))

print('Index of rows with Psi > 90: {}'.format(df[df.iloc[:, 2] > 90].index + 1))
