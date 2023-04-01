import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from Model import initial_data, extract_direction
import plotly.graph_objects as go

COMP_NAME = "icecube-neutrinos-in-deep-ice"
INPUT_PATH = Path(f"data/{COMP_NAME}")
OUTPUT_PATH = Path(f"data/{COMP_NAME}")
TRANSPARENCY_PATH = INPUT_PATH / "ice_transparency.txt"
_dtype = {
            "batch_id": "int16",
            "event_id": "int64",
        }
meta = pd.read_parquet(
    INPUT_PATH / f"train_meta.parquet"
).astype(_dtype)

train_batch_1 = pd.read_parquet(INPUT_PATH/f"train/batch_1.parquet")
sensor_geometry = pd.read_csv(INPUT_PATH/f"sensor_geometry.csv")

case_event_24 = train_batch_1.iloc[meta.loc[0].first_pulse_index.astype(int):meta.loc[0].last_pulse_index.astype(int)]

case_event_24 = case_event_24.set_index('sensor_id').join(sensor_geometry.set_index('sensor_id')[['x', 'y', 'z']]).reset_index()



# case_event_2092 = train_batch_1[train_batch_1.event_id == 2092 & train_batch_1.auxiliary == True]

batch_id = 1
event_id = 24

dataset = initial_data(batch_id)[event_id]
points = dataset.x.numpy()[:, :3]


fig_auxiliary = px.scatter_3d(
    case_event_24.loc[case_event_24.auxiliary],
    x='x', y='y', z='z', opacity=0.5, color_discrete_sequence=['red'])
vector_base = np.array([-500, 500])


azi, ele, center = extract_direction(batch_id, event_id)


import math

# x1 = center[0] * 500
# y1 = center[1] * 500
# z1 = center[2] * 500


# length = 250  # 直线长度
x = math.cos(math.radians(azi)) * math.sin(math.radians(ele)) * 500
y = math.sin(math.radians(azi)) * math.sin(math.radians(ele)) * 500
z = math.cos(math.radians(ele)) * 500

# x2, y2, z2 = x * length + x1, y * length + y1, z * length + z1
# x3, y3, z3 = x1 - x * length, y1 - y * length, z1 - z * length

x2 = math.cos(math.radians(azi)) * math.sin(math.radians(ele)) * -500
y2 = math.sin(math.radians(azi)) * math.sin(math.radians(ele)) * -500
z2 = math.cos(math.radians(ele)) * -500

df = pd.DataFrame({
    'x': [x, x2],
    'y': [y, y2],
    'z': [z, z2]
})
fig = px.line_3d(df, x='x', y='y', z='z')

fig_auxiliary.update_traces(marker_size=2)

figall = go.Figure(data = fig_auxiliary.data + fig.data)
figall.show()

# fig_auxiliary.show()
