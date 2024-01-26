import numpy as np
import pandas as pd

submissions_data = pd.read_csv('submissions_data_train.csv')

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
submissions_data['day'] = submissions_data.date.dt.date

users_scores = submissions_data.pivot_table(index='user_id',
                                            columns='submission_status',
                                            values='step_id',
                                            aggfunc='count',
                                            fill_value=0).reset_index()

events_data = pd.read_csv('event_data_train.csv')

events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
events_data['day'] = events_data.date.dt.date
gap_data = events_data[['user_id', 'day', 'timestamp']]\
    .drop_duplicates(subset=['user_id', 'day'])\
    .groupby('user_id')['timestamp'].apply(list)\
    .apply(np.diff).values

gap_data = pd.Series(np.concatenate(gap_data, axis=0))

gap_data.quantile(0.90) / (24*60*60)

users_events_data = events_data.pivot_table(index='user_id',
                                            columns='action',
                                            values='step_id',
                                            aggfunc='count',
                                            fill_value=0).reset_index()

users_data = events_data.groupby('user_id', as_index=False)\
    .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})

now = 1526772811
drop_out_threshold = 30 * 24 * 60 * 60
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold

users_data = users_data.merge(users_scores, on='user_id', how='outer')
users_data = users_data.fillna(0)
users_data = users_data.merge(users_events_data, how='outer')

users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()

users_data = users_data.merge(users_days, how='outer')
users_data['passed_course'] = users_data.passed > 170
# print(users_data[users_data.passed_course].day.median())
# users_data[users_data.passed_course].day.hist()

user_min_time = (events_data.groupby('user_id', as_index=False)
                 .agg({'timestamp': 'min'})
                 .rename({'timestamp': 'min_timestamp'}, axis=1))

users_data = users_data.merge(user_min_time, how='outer')

events_data['user_time'] = (events_data.user_id.map(str) + '_'
                            + events_data.timestamp.map(str))

learning_time_threshold = 3 * 24 * 60 * 60
user_learning_time_threshold = user_min_time.user_id.map(str) + '_' + \
                               (user_min_time.min_timestamp
                                + learning_time_threshold).map(str)
user_min_time['user_learning_time_threshold'] = user_learning_time_threshold

events_data = events_data.merge(user_min_time[['user_id',
                                               'user_learning_time_threshold']],
                                how='outer')

events_data_train = events_data[events_data.user_time
                                <= events_data.user_learning_time_threshold]

submissions_data['users_time'] = (submissions_data.user_id.map(str) + '_' +
                                  submissions_data.timestamp.map(str))
submissions_data = submissions_data.merge(user_min_time[['user_id',
                                                         'user_learning_time_threshold']],
                                          how='outer')
submissions_data_train = submissions_data[submissions_data.users_time <=
                                          submissions_data.user_learning_time_threshold]
