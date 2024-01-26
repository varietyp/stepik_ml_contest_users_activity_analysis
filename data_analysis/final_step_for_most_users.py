from users_actions_data import submissions_data

# finding final or the hardest step for most users

submissions = submissions_data.drop(['date', 'day'], axis=1)
worst_step = submissions[submissions.submission_status == 'wrong']\
    .groupby(['user_id', 'step_id'], as_index=False)\
    .agg({'timestamp': 'max'}).step_id.value_counts()

print(worst_step)
