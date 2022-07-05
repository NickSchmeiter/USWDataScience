
import matplotlib.pyplot as plt
from Regression.Regression_main import calcAccuracyDS1, calcAccuracyDS2, calcAccuracyDS3
from Classification.Classification_main import calcAccuracycDS1,calcAccuracycDS2,calcAccuracycDS3
fig, ax =plt.subplots(1,1)

data=[['Logistic Regression',
       calcAccuracyDS1(0), calcAccuracyDS2(0), calcAccuracyDS3(0)],
      ['Decision Tree', calcAccuracyDS1(1), calcAccuracyDS2(1), calcAccuracyDS3(1)],
      ['Random Forest', calcAccuracyDS1(2), calcAccuracyDS2(2), calcAccuracyDS3(2)],
      ['Linear SVR', calcAccuracyDS1(3), calcAccuracyDS2(3), calcAccuracyDS3(3)],
      ['Gradient Boosting', calcAccuracyDS1(4), calcAccuracyDS2(4), calcAccuracyDS3(4)]]
column_labels=["Regression", "ds wo outliers", "ds wo NaN", "ds wo NaN rpl by mean"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")
print("RegressionMatrix")
plt.show()


fig, ax =plt.subplots(1,1)

data=[['Logistic Regression',
       str(calcAccuracycDS1(0))+'%', str(calcAccuracycDS2(0))+'%', str(calcAccuracycDS3(0))+'%'],
      ['Decision Tree', str(calcAccuracycDS1(1))+'%', str(calcAccuracycDS2(1))+'%', str(calcAccuracycDS3(1))+'%'],
      ['Random Forest', str(calcAccuracycDS1(2))+'%', str(calcAccuracycDS2(2))+'%', str(calcAccuracycDS3(2))+'%'],
      ['Linear SVR', str(calcAccuracycDS1(3))+'%', str(calcAccuracycDS2(3))+'%', str(calcAccuracycDS3(3))+'%'],
      ['Gradient Boosting', str(calcAccuracycDS1(4))+'%', str(calcAccuracycDS2(4))+'%', str(calcAccuracycDS3(4))+'%']]
column_labels=["Classification", "ds wo outliers", "ds wo NaN", "ds wo NaN rpl by mean"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")
print("ClassificationMatrix")
plt.show()

