import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

class EvaluateModels:
    '''Base class to test text vectorizer and estimators in a pipeline using GridSearchCV.

    
    Parameters
    ----------
    
    vec: A list of vectorizers to evaluate. Either CountVectorizer or TfidfVectorizer or both
    models: A list of models to be evaluated
    params: A dictionary of parameters to be test using GridSearch. The 
            parameter key should start with the same str used for the 'name' 
            in the models tuple.
            Example: {'vectorizer': cvec, 'model__C': [0.001, 0.1, 1, 10, 100]}
    scoring: An sklearn scoring parameter
    '''

    def __init__(self, vecs, models, params: dict, scoring: str, num_folds: int = 5, seed: int = 23):
        self.vecs = vecs
        self.models = models
        self.params = params
        self.scoring = scoring
        self.num_folds = num_folds
        self.seed = seed
        self.grid = None
        self.best_params = None
        self.best_score = None
        self.best_pipe = None
        self.plot_labels = None
 
    def make_pipeline(self, vec, model):
        '''Make pipeline with vectorizer and model'''
        return Pipeline([('vectorizer', vec), ('model', model)])

    def get_best_results(self):
        '''Print the best score, the best set of grid parameters and the optimal model from Pipeline'''
        self.best_score = self.grid.best_score_
        print(f'The best mean cv score is {self.best_score}')
        self.best_params = self.grid.best_params_
        print(f'The best model parameters are {self.best_params}')
        self.best_pipe = self.grid.best_estimator_
        print(f'The best estimator pipeline is {self.best_pipe}')
    
    def get_plot_labels(self, result_df):
        col_labels = [col for col in result_df.columns if 'param_' in col]
        col_label = str(col_labels[0])
        plot_labels = result_df[col_label].values.tolist()
        return [str(i).split('(')[0] for i in plot_labels]
    
    def make_cvscore_df(self, result_df, plot_labels, clean_labels):
        score_cols = [col for col in result_df.columns if 'split' in col]
        cv_score_df = result_df[score_cols].T
        nums = range(len(plot_labels))
        label_mapper = dict(zip(nums, clean_labels))
        return cv_score_df.rename(columns= label_mapper)
    
    def plot_results(self):
        '''Plot the results of the key parameter from grid search results'''
        result_df = pd.DataFrame(self.grid.cv_results_)
        self.plot_labels = self.get_plot_labels(result_df)
        cv_score_df = self.make_cvscore_df(self.plot_labels)

        _, ax = plt.subplots()
        cv_score_df.boxplot()
        plt.ylabel("cv_score")
        ax.set_xticklabels(self.plot_labels)
        plt.show()
    
    def show_cvresults(self):
        print(pd.DataFrame(self.grid.cv_results_, 
                columns=['mean_train_score', 'mean_test_score', 'mean_fit_time', 'mean_score_time'], 
                index=self.plot_labels))
    
    # def make_gridsearch(self, pipe):
    #     kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
    #     grid = GridSearchCV(pipe, param_grid = self.params, cv=kfold, scoring= self.scoring, 
    #             # return_train_score=True
    #             )
    #     return grid
    
    def fit_pipeline(self, x_train, y_train):
        '''Creates pipeline and fits GridSearchCV with pipeline.
        Prints best results, plot of all tested parameters, 
        then prints table with cv results'''
        
        pipe = self.make_pipeline()
        # run grid search with pipeline
        # self.grid = self.make_gridsearch(pipe)
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        self.grid = GridSearchCV(pipe, param_grid = self.params, cv=kfold, scoring= self.scoring, 
                # return_train_score=True
                )
        self.grid.fit(x_train.values, y_train.values)
        self.get_best_results()
        self.plot_results()
        self.show_cvresults()
