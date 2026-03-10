from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


CLASSIFIERS = {
    'decision_tree': DecisionTreeClassifier,
    'linear_svc': LinearSVC,
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
}

PARAM_GRIDS = {
    'decision_tree': {
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__criterion': ['gini', 'entropy'],
    },
    'linear_svc': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__max_iter': [5000],
    },
    'random_forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    },
    'gradient_boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
    },
}


def get_pipeline(classifier_name: str, cv: int = 3, scoring: str = 'f1_macro') -> GridSearchCV:
    if classifier_name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(CLASSIFIERS.keys())}"
        )

    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', CLASSIFIERS[classifier_name]()),
    ])

    param_grid = {
        'selector__k': [10, 20, 'all'],
        **PARAM_GRIDS[classifier_name],
    }

    return GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=3)
