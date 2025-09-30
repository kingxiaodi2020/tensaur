import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
current_dir = os.getcwd()

go1 = pd.read_csv(current_dir + "/notebooks/go1_fixed_spine_euler.csv")
tens3_8000_10_RK4 = pd.read_csv(current_dir + "/notebooks/tens3_8000_10_rk4.csv") 
tens2_5000_12_5_euler = pd.read_csv(current_dir + "/notebooks/tens2_5000_12.5_euler.csv")
tens2_10000_25_euler = pd.read_csv(current_dir + "/notebooks/tens2_10000_25_euler.csv")

sns.set(style="darkgrid")

plt.figure(figsize=(10, 6))
plt.plot(go1['Step']*0.02, go1['vx (m/s)'], label='fixed_spine_euler', color='red')
plt.plot(tens3_8000_10_RK4['Step']*0.02, tens3_8000_10_RK4['vx (m/s)'], label='tens3_8000_10_RK4', color='blue')
plt.plot(tens2_5000_12_5_euler['Step']*0.02, tens2_5000_12_5_euler['vx (m/s)'], label='tens2_5000_12_5_euler', color='green')
plt.plot(tens2_10000_25_euler['Step']*0.02, tens2_10000_25_euler['vx (m/s)'], label='tens2_10000_25_euler', color='orange')
plt.plot(go1['Step']*0.02, np.ones(1000), label='command', color='black', linewidth=3)
plt.xlabel('Time (s)')
# plt.xlim(0, 5)
plt.ylabel('Velocity X (m/s)')
plt.legend(fontsize=12)
plt.title('Linear Velocity X Replayed')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(go1['Step']*0.02, go1['vy (m/s)'], label='fixed_spine_euler', color='red')
plt.plot(tens3_8000_10_RK4['Step']*0.02, tens3_8000_10_RK4['vy (m/s)'], label='tens3_8000_10_RK4', color='blue')
plt.plot(tens2_5000_12_5_euler['Step']*0.02, tens2_5000_12_5_euler['vy (m/s)'], label='tens2_5000_12_5_euler', color='green')
plt.plot(tens2_10000_25_euler['Step']*0.02, tens2_10000_25_euler['vy (m/s)'], label='tens2_10000_25_euler', color='orange')
plt.plot(go1['Step']*0.02, np.zeros(1000), label='command', color='black', linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Velocity Y (m/s)')
plt.legend(fontsize=12)
plt.title('Linear Velocity Y Replayed')
plt.tight_layout()
plt.show()