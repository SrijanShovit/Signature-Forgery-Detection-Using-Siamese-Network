import matplotlib.pyplot as plt

def metric_histogram(df, metric):

    plt.figure()
    df[metric].hist(bins=40)
    plt.title(metric)
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.show()


def classwise_boxplot(df, metric):

    if "label" not in df.columns:
        return

    plt.figure()
    df.boxplot(column=metric, by="label")
    plt.title(metric)
    plt.show()