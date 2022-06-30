
import matplotlib.pyplot as plt
from Regression.Regression_main import calcAccuracyDS1, calcAccuracyDS2, calcAccuracyDS3

fig, ax =plt.subplots(1,1)

data=[['Logistic Regression',
       calcAccuracyDS1(0), calcAccuracyDS2(0), calcAccuracyDS3(0)],
      ['Decision Tree', calcAccuracyDS1(1), calcAccuracyDS2(1), calcAccuracyDS3(1)],
      ['Random Forest', calcAccuracyDS1(2), calcAccuracyDS2(2), calcAccuracyDS3(2)],
      ['Linear SVR', calcAccuracyDS1(3), calcAccuracyDS2(3), calcAccuracyDS3(3)],
      ['Gradient Boosting', calcAccuracyDS1(4), calcAccuracyDS2(4), calcAccuracyDS3(4)]]
column_labels=[" ", "ds wo outliers", "ds wo NaN", "ds wo NaN rpl by mean"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")

plt.show()

