from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from scipy.stats.stats import pearsonr
import os
from matplotlib import pyplot as plt


def run_lasso_main(X_tr, y_tr, X_vl, y_val, output, main):
    os.chdir(output)

    if not os.path.exists("lasso"):
        os.makedirs("lasso")
        dir = output + "lasso/"

    os.chdir(os.path.join(output, 'lasso/'))

    if not os.path.exists("figures"):
        os.makedirs("figures")
        dir = output + "lasso/figures/"

    os.chdir(os.path.join(output, 'lasso/figures/'))

    for i in range(0, y_tr.shape[1]):
        lasso = LassoCV(cv=5, random_state=0).fit(X_tr, y_tr[:,i])
        # lasso = linear_model.Lasso(alpha=0.01)
        # lasso.fit(X_train, y_train)
        lasso.score(X_tr, y_tr[:,i])
        y_hat = lasso.predict(X_vl)

        # correlation btw predicted and observed
        corr = pearsonr(y_hat, y_val[:,i])
        fig = plt.figure()
        # plot observed vs. predicted targets
        plt.title('Lasso: Observed vs Predicted Y_trait_' + str(i) + 'cor:' + str(corr[0]))
        plt.ylabel('Predicted')
        plt.xlabel('Observed')
        plt.scatter(y_val[:,i], y_hat, marker='o')
        fig.savefig("Lasso_Out" + str(i) + '.png', dpi=300)
        plt.close(fig)

