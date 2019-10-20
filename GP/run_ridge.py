from sklearn.linear_model import RidgeCV
from scipy.stats.stats import pearsonr
import os
from matplotlib import pyplot as plt


def run_ridge_main(X_tr, y_tr, X_vl, y_val, output, main):

    os.chdir(output)

    if not os.path.exists("ridge"):
            os.makedirs("ridge")
            dir = output + "ridge/"

    os.chdir(os.path.join(output, 'ridge/'))


    if not os.path.exists("figures"):
        os.makedirs("figures")
        dir = output + "ridge/figures/"

    os.chdir(os.path.join(output, 'ridge/figures/'))

    for i in range(0, y_tr.shape[1]):

        ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_tr, y_tr[:,i])
        ridge.score(X_tr, y_tr[:, i])
        y_hat = ridge.predict(X_vl)


        # correlation btw predicted and observed
        corr = pearsonr(y_hat, y_val[:,i])
        fig = plt.figure()
        # plot observed vs. predicted targets
        plt.title('Ridge: Observed vs Predicted Y_trait_' + str(i) + 'cor:' + str(corr[0]))
        plt.ylabel('Predicted')
        plt.xlabel('Observed')
        plt.scatter(y_val[:,i], y_hat, marker='o')
        fig.savefig("Ridge_Out" + str(i) + '.png', dpi=300)
        plt.close(fig)


