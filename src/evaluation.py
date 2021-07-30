import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def classification_report(gt, p):
    fpr, tpr, thresholds = roc_curve(y_true=gt, y_score=p)
    df_roc = pd.DataFrame(data={"Fpr": fpr,
                                "Tpr": tpr,
                                "Thresholds": thresholds})
    df_roc.to_csv(file + "csv", index=False, header=True)
    df_roc.plot(x='Fpr', y='Tpr', title="ROC ", kind='line')
    plt.savefig(file + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()