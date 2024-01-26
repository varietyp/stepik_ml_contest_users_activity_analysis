import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from users_actions_data import events_data

sns.set(rc={'figure.figsize': (9, 6)})

events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date
events_data.groupby('day').user_id.nunique().plot()
plt.show()