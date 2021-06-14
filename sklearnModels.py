import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas
import sys

from sklearn.linear_model import LinearRegression

# Usage:
# python3 sklearnModels.py BlackWhiteVotingPercentageUS.csv "Year Voted" "Inequity (%)"

def main():
    filename = sys.argv[1]
    col1 = sys.argv[2]
    col2 = sys.argv[3]
    df = pandas.read_csv(filename)
    df = df[[col1, col2]]
    x = df[col1].to_numpy()
    x = np.reshape(x, (-1, 1))
    train_stop = len(x) // 2
    x_train = x[train_stop : ]
    x_test = x[ : train_stop]
    y = df[col2]
    y_train = y[train_stop : ]
    y_test = y[ : train_stop]
    reg = LinearRegression().fit(x_train, y_train)
    y_hat = reg.predict(x_test)
    print(x_test)
    print(y_hat)
    print(y_test)
    plot_voting_gap()

def plot_voting_gap():
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    # The first row is garbage
    # The 0 column is "year from start"
    # The 1 column is the equity gap
    fname    = 'BlackWhiteVotingPercentageUS.csv'
    data_in  = np.loadtxt(fname, delimiter=',', skiprows=4, usecols=(0,1,2,3))
    rows     = data_in.shape[0]
    print(rows)

    # Keep track of the earliest year; make it year 0; each row is two years.
    year0    = data_in[-1,0]    # Last year in the data set
    print(year0)

    # N is what is used to plot against on the y-axis.
    # x is the data being predicted.
    # Flip is because I know this data set is in reverse time order.
    N        = np.flip(data_in[0:, 0])

    # Ratio minus 1 (equality)
    x        = np.flip(data_in[0:, 2] / data_in[0:,3]) - 1.0
    print(x)
    plt.ion()
    plt.figure(1)

    plt.plot(N, x + 1.0, 'g-o', linewidth=2, label='Real data')
    plt.plot([min(N)-0.5, max(N)+0.5], [1, 1], 'k-', linewidth=1)
    plt.xlim(min(N)-0.5, max(N)+0.5)
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Inequality Ratio", fontsize=20)
    plt.xticks(range(int(min(N)), int(max(N)), 8))
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()
    plt.savefig("basedata_votinggap_ratio.png")
    #plt.legend(loc='lower left', fontsize=14)
    #plt.ylim(0.8, 1.5)
    #plt.show()


if __name__ == "__main__":
    main()
