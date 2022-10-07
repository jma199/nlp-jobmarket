import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import KFold

class EvaluateModels:
    '''Base class to test text vectorizer and estimators in a pipeline using GridSearchCV.
    
    Parameters
    ----------
    
    models: A list of tuples of estimators to be evaluated
    constant_model: use either "vectorizer" or "estimator"
    constant_model_name: A tuple containing ('model name', model()) to be evaluated
    params: A dictionary of parameters to be test using GridSearch. The 
            parameter key should start with the same str used for the 'name' 
            in the models tuple.
            Example: {'vectorizer': cvec, 'model__C': [0.001, 0.1, 1, 10, 100]}
    scoring: An sklearn scoring parameter
    '''

    def __init__(self, model_list, constant_model, test_type:str, scoring: str, params=None, num_folds: int = 5, seed: int = 23):
        if params is None:
            params = {}
        self.model_list = model_list
        self.constant_model = constant_model
        self.test_type = test_type
        self.scoring = scoring
        self.num_folds = num_folds
        self.seed = seed

        self.params = params
        self.cv_search = None
        self.best_score = None
        self.best_model = None
        self.best_params = None
        self.plot_labels = None
 
    def make_pipeline(self, model):
        '''Make pipeline with vectorizer and model'''
        if self.test_type == 'vectorizer':
            model_pipe = Pipeline([model, self.constant_model])           
        if self.test_type == 'estimator':
            model_pipe = Pipeline([self.constant_model, model]) 
        return model_pipe

    def store_best_results(self, model):
        '''Print the best cv test score and the optimal model'''
        if not self.best_score:
            self.best_score = np.mean(self.cv_search['test_score'])
            self.best_model = model
        elif np.mean(self.cv_search['test_score']) > self.best_score:
            self.best_score = np.mean(self.cv_search['test_score'])
            self.best_model = model
    
    def plot_testscores_box(self, test_scores, cv_index):
        '''Plot the test scores from cross_validate for each model as a boxplot'''
        nums = range(len(test_scores))
        label_mapper = dict(zip(nums, cv_index))
        data = pd.DataFrame(test_scores).T.rename(columns=label_mapper)
        # print('Dataframe: Test Scores for each Model Tested')
        # print(data)
        print('='*50)
        data.plot(kind='box', xlabel=cv_index)
        plt.ylabel("cv_score")
        plt.xticks(rotation=30)
        plt.show()
    
    def get_plot_labels(self, result_df):
        '''From GridSearchCV results, get list with x labels'''
        col_labels = [col for col in result_df.columns if 'param_' in col]
        col_label = str(col_labels[0])
        plot_labels = result_df[col_label].values.tolist()
        return [str(i).split('(')[0] for i in plot_labels]
    
    def make_cvscore_df(self, result_df, plot_labels, clean_labels):
        '''From GridSearchCV results, get test scores to plot'''
        score_cols = [col for col in result_df.columns if 'split' in col]
        cv_score_df = result_df[score_cols].T
        nums = range(len(plot_labels))
        label_mapper = dict(zip(nums, clean_labels))
        return cv_score_df.rename(columns= label_mapper)
    
    def print_best_cvresults(self):
        print('='*40)
        print(f'The best mean test score: {self.best_score}')
        print(f'The best model: {self.best_model}', end='\n')
    
    def print_cvresults(self, cv_results, cv_index):
        '''Get cross_validate results'''
        self.result_df = (pd.DataFrame(data = cv_results, index= cv_index,
                columns=['test_score', 'train_score', 'fit_time', 'score_time'], 
                ))
        print('='*40)
        print(self.result_df)

    # def plot_mean_results(self):
    #     '''Plot the results of the key parameter from grid search results
    #       Plots bar graph using mean value of scoring parameter.'''
    #     plot_data = self.result_df['test_score']
    #     plot_data.plot(kind='bar', use_index=True)
    #     plt.ylabel("cv_score")
    #     plt.show()
    
    def run(self, x_train, y_train):
        '''Creates pipeline and fits GridSearchCV with pipeline.
        Prints best results, plot of all tested parameters, 
        then prints table with cv results'''

        # if self.test_type == 'pipeline':
            # model = model_list[1]
        
        # run grid search with pipeline
        X = x_train.values
        y = y_train.values

        cv_results = [] # mean model scores
        cv_std = []
        cv_index = [] # list of models
        test_scores = [] # list of model test scores

        for model in self.model_list: # for each tuple in list of tuples
            print(f'Testing {model[0]}')
            model_pipe = self.make_pipeline(model)
            
            # check = [key for key in self.params.keys() if model[0]]
            # if len(check) == 1:
            # # if model[0] in self.params.keys():
            #     model = self.tune_hyperparameters(model, X,  y)
            
            kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
            self.cv_search = cross_validate(model_pipe, X, y, cv=kfold, scoring= self.scoring, return_train_score=True)        

            # get scores
            scores = {name:np.mean(value) for name, value in self.cv_search.items()}
            cv_results.append(scores)
            std = {name:np.std(value) for name, value in self.cv_search.items()}
            cv_std.append(std)
            cv_index.append(model[0])

            # store individual test scores
            model_test_scores = self.cv_search['test_score']
            test_scores.append(model_test_scores)
            
            # store best results
            self.store_best_results(model)

        self.print_best_cvresults()
        self.print_cvresults(cv_results, cv_index)
        self.plot_testscores_box(test_scores, cv_index)

    def tune_hyperparameters(self, model, x_train, y_train):
        '''Perform hyperparameter tuning and set best parameter for best performing model'''
        pipe = self.make_pipeline(model)
        kfold = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        self.grid_search = GridSearchCV(pipe, 
                                        param_grid = self.params, 
                                        cv=kfold, 
                                        scoring= self.scoring, 
                                        return_train_score=True)
        self.grid_search.fit(x_train.values, y_train.values)
        
        print(f'Best parameters found for {model[0]}:')
        print(f'  -- {self.grid_search.best_params_}')
        model.set_params(**self.grid_search.best_params_)

class EvaluatePreprocessors(EvaluateModels):
    '''Subclass that evaluates multiple preprocessors in a pipeline with one estimator'''
    def __init__(self, preprocessors: list, estimator, test_type, scoring, **kwargs):
        super().__init__(preprocessors, estimator, test_type, scoring=scoring, **kwargs)

class EvaluateEstimators(EvaluateModels):
    '''Subclass that evaluates multiple estimators in a pipeline with one preprocessor'''
    def __init__(self, estimators: list, preprocessor, test_type, scoring, **kwargs):
        super().__init__(estimators, preprocessor, test_type, scoring=scoring, **kwargs)
