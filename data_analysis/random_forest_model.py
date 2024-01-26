import pandas as pd

from users_actions_data import events_data_train, submissions_data_train, users_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index()\
    .rename(columns={'day': 'days'})
steps_tried = submissions_data_train.groupby('user_id').step_id\
    .nunique().to_frame().reset_index().rename(columns={'step_id': 'steps_tried'})
X = X.merge(steps_tried, on='user_id', how='outer')
users_scores = submissions_data_train.pivot_table(index='user_id',
                                                  columns='submission_status',
                                                  values='step_id',
                                                  aggfunc='count',
                                                  fill_value=0).reset_index()
X = X.merge(users_scores)
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X = X.merge(events_data_train.pivot_table(index='user_id',
                                          columns='action',
                                          values='step_id',
                                          aggfunc='count',
                                          fill_value=0)\
            .reset_index()[['user_id', 'viewed']], how='outer')

X = X.fillna(0)
X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')
X = X[~((X.passed_course == False) & (X.is_gone_user == False))]

y = X.passed_course.map(int)
X = X.drop(['passed_course', 'is_gone_user'], axis=1)
X = X.set_index(X.user_id)
X = X.drop('user_id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33, random_state=42)

# clf_rf = RandomForestClassifier()

parameters = {'n_estimators': [25, 50, 100, 150],
              'max_features': ['auto', 'sqrt'],
              'max_depth': [int(x) for x in np.linspace(10, 120, num=12)],
              'min_samples_split': [2, 6, 10],
              'min_samples_leaf': [1, 3, 4],
              'bootstrap': [True, False]
}

# grid_search_cv_clf = GridSearchCV(clf_rf, parameters)
# grid_search_cv_clf.fit(X_train, y_train)
# print(grid_search_cv_clf.best_estimator_)

# model_rf = RandomForestClassifier(max_depth=9, max_features='log2', max_leaf_nodes=9,
#                                   n_estimators=50)
# model_rf.fit(X_train, y_train)
# y_pred_rand = model_rf.predict(X_test)
# print(classification_report(y_pred_rand, y_test))

# model_rf.fit(X_train, y_train)
# pred_proba = model_rf.predict_proba(X_test)
# roc_score = roc_auc_score(y_test, pred_proba[:, 1])
# print(roc_score)

# random_search_model = RandomizedSearchCV(RandomForestClassifier(), parameters)
# random_search_model.fit(X_train, y_train)
# print(random_search_model.best_estimator_)

# model_random = RandomForestClassifier(max_depth=3, max_features=None, max_leaf_nodes=6,
#                                      n_estimators=100)
# model_random.fit(X_train, y_train)
# pred_proba = model_random.predict_proba(X_test)
# roc_score = roc_auc_score(y_test, pred_proba[:, 1])
# print(roc_score)

# random_rf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=1)
# random_rf.fit(X_train, y_train)
# print(random_rf.best_estimator_)

model_rf = RandomForestClassifier(max_depth=90, max_features='sqrt', min_samples_leaf=4,
                                  n_estimators=25)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
y_pred_rf = pd.DataFrame({'actual': y_test, 'pred_proba': y_pred})

print(y_pred_rf)

pred_proba = model_rf.predict_proba(X_test)
roc_score = roc_auc_score(y_test, pred_proba[:, 1])
print(roc_score)