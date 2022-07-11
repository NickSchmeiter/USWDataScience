
import matplotlib.pyplot as plt
from Regression.Regression_main import calcAccuracyDS1, calcAccuracyDS2, calcAccuracyDS3, calcAccuracyDS4
from Classification.Classification_main import calcAccuracycDS1,calcAccuracycDS2,calcAccuracycDS3

fig, ax =plt.subplots(1,1)

#build accuracy matrix for regression
data=[[
       str(calcAccuracyDS1(0)) + ' %', str(calcAccuracyDS2(0)) + ' %', str(calcAccuracyDS3(0)) + ' %', str(calcAccuracyDS4(0)) + ' %'],
      [str(calcAccuracyDS1(1))+ ' %', str(calcAccuracyDS2(1))+ ' %', str(calcAccuracyDS3(1))+ ' %', str(calcAccuracyDS4(1))+' %'],
      [str(calcAccuracyDS1(2))+ ' %', str(calcAccuracyDS2(2))+ ' %', str(calcAccuracyDS3(2))+ ' %', str(calcAccuracyDS4(2))+ ' %'],
      [str(calcAccuracyDS1(3))+ ' %', str(calcAccuracyDS2(3))+ ' %', str(calcAccuracyDS3(3))+ ' %', str(calcAccuracyDS4(3))+ ' %'],
      [str(calcAccuracyDS1(4))+ ' %', str(calcAccuracyDS2(4))+ ' %', str(calcAccuracyDS3(4))+ ' %', str(calcAccuracyDS4(4))+ ' %']]

column_labels=["Regression", "ds wo outliers", "ds wo NaN", "ds wo NaN rpl by mean", "ds wo NaN IQR"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")
print("RegressionMatrix")
plt.show()

#build accuracy matrix for classification
fig, ax =plt.subplots(1,1)

data=[[
       str(calcAccuracycDS1(0))+'%', str(calcAccuracycDS2(0))+'%', str(calcAccuracycDS3(0))+'%', str(calcAccuracyDS4(0)) + ' %'],
      [str(calcAccuracycDS1(1))+'%', str(calcAccuracycDS2(1))+'%', str(calcAccuracycDS3(1))+'%', str(calcAccuracyDS4(1)) + ' %'],
      [ str(calcAccuracycDS1(2))+'%', str(calcAccuracycDS2(2))+'%', str(calcAccuracycDS3(2))+'%', str(calcAccuracyDS4(2)) + ' %'],
      [str(calcAccuracycDS1(3))+'%', str(calcAccuracycDS2(3))+'%', str(calcAccuracycDS3(3))+'%', str(calcAccuracyDS4(3)) + ' %'],
      [str(calcAccuracycDS1(4))+'%', str(calcAccuracycDS2(4))+'%', str(calcAccuracycDS3(4))+'%', str(calcAccuracyDS4(4)) + ' %']]

column_labels=["ds wo outliers", "ds wo NaN", "ds wo NaN rpl by mean","ds wo NaN IQR"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")
print("ClassificationMatrix")
plt.show()


