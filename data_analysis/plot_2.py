import matplotlib.pyplot as plt
import seaborn as sns
from users_actions_data import events_data

sns.set(rc={'figure.figsize': (9, 6)})

actions_data = events_data.pivot_table(index='user_id',
                                       columns='action', values='step_id',
                                       aggfunc='count', fill_value=0).reset_index()
actions_data.discovered.hist()
plt.show()