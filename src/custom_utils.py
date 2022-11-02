from sklearn.model_selection import learning_curve, validation_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(classifier, X, y, cv=None, scoring=None, figsize=None):
    """Plot the learning curve"""

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(classifier, X, y, return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot learning curve
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="CV Score")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
    
    ax.set_xlabel("Train size")
    ax.set_ylabel("scoring")

    plt.legend(loc="best")
    plt.title("Learning Curve")
    plt.show()
    

def plot_validation_curve(estimator, X, y, param_name = None, param_range=None, scoring= None, fig_size=None, semilog=False, **kwargs):
    """Plot validation curve.
    Validation curve is used to determine the train and test scores for varying parameter values"""   
    train_scores, test_scores = validation_curve(estimator, X, y, param_name = param_name, param_range = param_range, scoring=scoring, **kwargs)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    test_score_dict = dict(zip(param_range, test_scores_mean))
    m = max(test_score_dict, key=test_score_dict.get)
    print(f"Parameter with best CV score: {m}, {test_score_dict[m]}")
    print(f"Best CV score: {test_score_dict[m]}")

    _, ax = plt.subplots(figsize=fig_size)

    if semilog:
        ax.semilogx(param_range, train_scores_mean, 'o-', label='Train', color='b')
        ax.semilogx(param_range, test_scores_mean, 'o-', label='Cross Validation', color='darkorange')
    else:
        ax.plot(param_range, train_scores_mean, 'o-', label='Train', color='b')
        ax.plot(param_range, test_scores_mean, 'o-', label='Cross Validation', color='darkorange')
    
    ax.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='b')
    ax.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='darkorange')
        
    ax.set_xlabel(param_name)
    ax.set_ylabel("scoring")

    plt.legend(loc="best")
    plt.title("Validation Curve")
    plt.show()
