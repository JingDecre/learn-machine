import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("cs_training.csv").drop("Unnamed: 0", axis=1)

# 使用 describe() 来看数据集中的计数、均值、最大最小值、标准差和第一、二、三个四分位值，同时增加了缺失率的计算
df.describe().T.assign(missing_rate = df.apply(lambda x : (len(x)-x.count())/float(len(x)))).to_csv('describe_total.csv')

# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.iloc[:, [5,0,1,2,3,4,6,7,8,9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].values # 转成数组
    unknown = process_df[process_df.MonthlyIncome.isnull()].values
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print("预测值： ", predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df

df = set_missing(df)
df = df.dropna() # 删除缺失值
df = df.drop_duplicates() # 删除重复值

###########异常值处理
#异常值：偏离大多数抽样数据的数值，通常指测定值中与平均值的偏差超过两倍标准差的测定值
# 通常采用离群值检测的方法对异常值进行检测
# 使用2来代替大于2的值
revNew = []
for val in df.RevolvingUtilizationOfUnsecuredLines:
    if val <= 2:
        revNew.append(val)
    else:
        revNew.append(2.)
df.RevolvingUtilizationOfUnsecuredLines = revNew

# df["RevolvingUtilizationOfUnsecuredLines"].plot(kind="box",grid=True)

# df.age.plot.box(grid=True)
# 发现 age 属性中存在0值情况，而这些数据明显是异常值，因此对其进行处理
df = df[df["age"] > 0]

# df.boxplot(column=["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"],
#             rot=30)
# print("NumberOfTime30-59DaysPastDueNotWorse:", df["NumberOfTime30-59DaysPastDueNotWorse"].unique())
# print("NumberOfTime60-89DaysPastDueNotWorse:", df["NumberOfTime60-89DaysPastDueNotWorse"].unique())
# print("NumberOfTimes90DaysLate:", df["NumberOfTimes90DaysLate"].unique())

def replaceOutlier(data):
    New = []
    med = data.median()
    for val in data:
        if ((val == 98) | (val == 96)):
            New.append(med)
        else:
            New.append(val)
    return New

df["NumberOfTime30-59DaysPastDueNotWorse"] = replaceOutlier(df["NumberOfTime30-59DaysPastDueNotWorse"])
df["NumberOfTime60-89DaysPastDueNotWorse"] = replaceOutlier(df["NumberOfTime60-89DaysPastDueNotWorse"])
df["NumberOfTimes90DaysLate"] = replaceOutlier(df["NumberOfTimes90DaysLate"])

# df["DebtRatio"].plot(kind="box")

# 使用中位数绝对偏差 MAD（median absolute deviation）方法进行异常值的检测
from scipy.stats import norm

def mad_based_outlier(points, thresh=3.5):
    if type(points) is list:
        points = np.asarray(points)
    if len(points.shape) == 1:
        points = points[:, None]
    med = np.median(points, axis=0)
    abs_dev = np.absolute(points - med)
    med_abs_dev = np.median(abs_dev)

    mod_z_score = norm.ppf(0.75) * abs_dev / med_abs_dev
    return mod_z_score > thresh

# 检测出最小的异常值,用于替换异常值
minUpperBound = min([val for (val, out) in zip(df.DebtRatio, mad_based_outlier(df.DebtRatio)) if out == True]) # zip将多个列表组成新的列表


newDebtRatio = []
for val in df.DebtRatio:
    if val > minUpperBound:
        newDebtRatio.append(minUpperBound)
    else:
        newDebtRatio.append(val)

df.DebtRatio = newDebtRatio

# df.DebtRatio.describe()

minUpperBound_MonthlyIncome = min([val for (val, out) in zip(df.MonthlyIncome, mad_based_outlier(df.MonthlyIncome)) if out == True])

newMonthlyIncome = []
for val in df.MonthlyIncome:
    if val > minUpperBound_MonthlyIncome:
        newMonthlyIncome.append(minUpperBound_MonthlyIncome)
    else:
        newMonthlyIncome.append(val)

df.MonthlyIncome = newMonthlyIncome

# df.MonthlyIncome.plot.box()


minUpperBound_NumberOfOpenCreditLinesAndLoans = min([val for (val, out) in zip(df.NumberOfOpenCreditLinesAndLoans, mad_based_outlier(df.NumberOfOpenCreditLinesAndLoans)) if out == True])

newNumberOfOpenCreditLinesAndLoans = []
for val in df.NumberOfOpenCreditLinesAndLoans:
    if val > minUpperBound_NumberOfOpenCreditLinesAndLoans:
        newNumberOfOpenCreditLinesAndLoans.append(minUpperBound_NumberOfOpenCreditLinesAndLoans)
    else:
        newNumberOfOpenCreditLinesAndLoans.append(val)

df.NumberOfOpenCreditLinesAndLoans = newNumberOfOpenCreditLinesAndLoans

# df.NumberOfOpenCreditLinesAndLoans.describe()
# df["NumberOfOpenCreditLinesAndLoans"].plot(kind="box", grid=True)

realNew = []
for val in df.NumberRealEstateLoansOrLines:
    if val > 17:
        realNew.append(17)
    else:
        realNew.append(val)

df.NumberRealEstateLoansOrLines = realNew

# df["NumberRealEstateLoansOrLines"].plot(kind="box", grid=True)

depNew = []
for var in df.NumberOfDependents:
    if var > 10:
        depNew.append(10)
    else:
        depNew.append(var)

df.NumberOfDependents = depNew

# df.NumberOfDependents.plot.box(grid=True)


#######数据切分
from sklearn.model_selection import train_test_split

Y = df["SeriousDlqin2yrs"]
X = df.iloc[:, 1:]
# 测试和训练数据进行3：7的比例进行切分 random_state定一个值是的每次运行的时候不会被随机分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)

train.to_csv('TrainData.csv',index=False)
test.to_csv('TestData.csv',index=False)


#######探索性数据分析

# 下面利用直方图和核密度估计画图，Age、MonthlyIncome、NumberOfOpenCreditLinesAndLoans大致呈正太分布，符合统计分析

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2, 3), (0, 0))
train["age"].plot(kind="hist", bins=30, figsize=(12, 6), grid=True)
plt.title("Hist of Age")

plt.subplot2grid((2, 3), (0, 1))
train["age"].plot(kind="kde", grid=True)
plt.title("KDE of Age")

plt.subplot2grid((2, 3), (0, 2))
train["MonthlyIncome"].plot(kind="kde", grid=True)
plt.xlim(-20000, 80000)
plt.title("KDE of MonthlyIncome")

plt.subplot2grid((2, 3), (1, 0))
train["NumberOfDependents"].plot(kind="kde")
plt.title("KDE of NumberOfDependents")

plt.subplot2grid((2, 3), (1, 1))
train["NumberOfOpenCreditLinesAndLoans"].plot(kind="kde")
plt.title("KDE of NumberOfOpenCreditLinesAndLoans")

plt.subplot2grid((2, 3), (1, 2))
train["NumberRealEstateLoansOrLines"].plot(kind="kde")
plt.title("KDE of NumberRealEstateLoansOrLines")

# 解决中文的显示问题
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.tight_layout() # 调整子图之间的间距，紧凑显示图像
plt.show()

# 特征选择
# 变量分箱：
#
#     将连续变量离散化
#
#     将多状态的离散变量合并成少状态
#
# 变量分箱的重要性：
#
#     1、稳定性：避免特征中无意义的波动对评分带来波动
#
#     2、健壮性：避免极端值的影响
#
#  变量分箱的优势：
#
#      1、可以将缺失值作为一个独立的箱带入模型中
#
#      2、将所有的变量变换到相似的尺度上
#
#  变量分箱的劣势：
#
#      1、计算量大
#
#      2、分箱之后需要编码
#
#  变量分箱常用的方法：
#
#      有监督的：
#
#         1、Best-KS； 2、ChiMerge（卡方分箱法）
#
#      无监督的：
#
#         1、等距； 2、等频； 3、聚类
import scipy.stats.stats as stats

# 自定义自动分箱函数
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X.rank(method="first"), n)}) # X.rank(method="first")
        d2 = d1.groupby("Bucket", as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y) # 使用斯皮尔曼等级相关系数来评估两个变量之间的相关性
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1-d3['rate'])) / (good/bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min'))
    print(d4)
    cut=[]
    cut.append(float('-inf'))
    for i in range(1, n+1):
        qua = X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe

# 自定义分箱函数
def self_bin(Y, X, cat):
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({'X': X, 'Y': Y,'Bucket': pd.cut(X, cat)})
    d2 = d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min'))
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, iv, woe

dfx1, ivx1, cutx1, woex1 = mono_bin(train.SeriousDlqin2yrs, train.RevolvingUtilizationOfUnsecuredLines, n=10)

dfx2, ivx2, cutx2, woex2 = mono_bin(train.SeriousDlqin2yrs, train.age, n=10)

pinf = float('inf') # 正无穷大
ninf = float('-inf') # 负无穷大

cutx3 = [ninf, 0, 1, 3, 5, pinf]
dfx3, ivx3, woex3 = self_bin(train.SeriousDlqin2yrs, train["NumberOfTime30-59DaysPastDueNotWorse"], cutx3)

# df.DebtRatio.describe()
dfx4, ivx4, cutx4, woex4 = mono_bin(train.SeriousDlqin2yrs, train.DebtRatio, n=20)

dfx5, ivx5, cutx5, woex5 = mono_bin(train.SeriousDlqin2yrs, train.MonthlyIncome, n=10)

cutx6 = [ninf, 1, 2, 3, 5, pinf]
dfx6, ivx6, woex6 = self_bin(train.SeriousDlqin2yrs, train.NumberOfOpenCreditLinesAndLoans, cutx6)

cutx7 = [ninf, 0, 1, 3, pinf]
dfx7, ivx7, woex7 = self_bin(train.SeriousDlqin2yrs, train["NumberOfTimes90DaysLate"], cutx7)

cutx8 = [ninf, 0, 1, 2, 3, pinf]
dfx8, ivx8, woex8 = self_bin(train.SeriousDlqin2yrs, train["NumberRealEstateLoansOrLines"], cutx8)

cutx9 = [ninf, 0, 1, 3, pinf]
dfx9, ivx9, woex9 = self_bin(train.SeriousDlqin2yrs, train["NumberOfTime60-89DaysPastDueNotWorse"], cutx9)

cutx10 = [ninf, 0, 1, 2, pinf]
dfx10, ivx10, woex10 = self_bin(train.SeriousDlqin2yrs, train["NumberOfDependents"], cutx10)


corr = train.corr() # 计算各变量的相关性系数
xticks = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'] # x轴标签
yticks = list(corr.index) # y轴标签
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax,
            annot_kws={'size': 12, 'weight': 'bold', 'color': 'blue'}) # 绘制相关性系数热力图
ax.set_xticklabels(xticks, rotation=0, fontsize=12)
ax.set_yticklabels(yticks, rotation=0, fontsize=12)
plt.show()

# 计算iv值
ivlist = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10] # 各变量IV
index = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'] # x轴的标签
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index)) + 1
ax1.bar(x, ivlist, width=0.4) # 生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=12)
# 在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()


# 替换成woe函数
def replace_woe(series, cut, woe):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut)-2
        m = len(cut)-2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(woe[m])
        i += 1
    return list


from pandas import Series
train = pd.read_csv("TrainData.csv")
train['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(train['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
train['age'] = Series(replace_woe(train['age'], cutx2, woex2))
train['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(train['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
train['DebtRatio'] = Series(replace_woe(train['DebtRatio'], cutx4, woex4))
train['MonthlyIncome'] = Series(replace_woe(train['MonthlyIncome'], cutx5, woex5))
train['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(train['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
train['NumberOfTimes90DaysLate'] = Series(replace_woe(train['NumberOfTimes90DaysLate'], cutx7, woex7))
train['NumberRealEstateLoansOrLines'] = Series(replace_woe(train['NumberRealEstateLoansOrLines'], cutx8, woex8))
train['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(train['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
train['NumberOfDependents'] = Series(replace_woe(train['NumberOfDependents'], cutx10, woex10))
train.to_csv('Woetrain.csv', index=False)

test = pd.read_csv('TestData.csv')
# 替换成woe
test['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
test['age'] = Series(replace_woe(test['age'], cutx2, woex2))
test['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
test['DebtRatio'] = Series(replace_woe(test['DebtRatio'], cutx4, woex4))
test['MonthlyIncome'] = Series(replace_woe(test['MonthlyIncome'], cutx5, woex5))
test['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
test['NumberOfTimes90DaysLate'] = Series(replace_woe(test['NumberOfTimes90DaysLate'], cutx7, woex7))
test['NumberRealEstateLoansOrLines'] = Series(replace_woe(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
test['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
test['NumberOfDependents'] = Series(replace_woe(test['NumberOfDependents'], cutx10, woex10))
test.to_csv('TestWoeData.csv', index=False)


#################模型预测
from sklearn.model_selection import cross_val_score


def cvDictGen(functions, scr, X_train=X_train, Y_train=Y_train, cv=10, verbose=1):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, Y_train, cv=cv, verbose=verbose, scoring=scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]

    return cvDict


def cvDictNormalize(cvDict):
    cvDictNormalized = {}
    for key in cvDict.keys():
        for i in cvDict[key]:
            cvDictNormalized[key] = ['{:0.2f}'.format((cvDict[key][0] / cvDict[list(cvDict.keys())[0]][0])),
                                     '{:0.2f}'.format((cvDict[key][1] / cvDict[list(cvDict.keys())[0]][1]))]
    return cvDictNormalized

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

knMod = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                             metric='minkowski', metric_params=None)

lrMod = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100,
                            multi_class='ovr', verbose=2)
adaMod = AdaBoostClassifier(base_estimator=None, n_estimators=200, learning_rate=1.0)

gbMod = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200, subsample=1.0,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                                   init=None, random_state=None, max_features=None, verbose=0)

rfMod = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0)

cvD = cvDictGen(functions=[knMod, lrMod, adaMod, gbMod, rfMod], scr='roc_auc')

# print(cvD)

cvDictNormalize(cvD)

#########最优化超参数
##AdaBoost模型


from sklearn.model_selection import RandomizedSearchCV
from random import randint


ada_param = {'n_estimators': [10,50,100,200,400],
                 'learning_rate': [0.1, 0.05]}

randomizedSearchAda = RandomizedSearchCV(estimator=adaMod, param_distributions=ada_param, n_iter=5,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, Y_train)

# randomizedSearchAda.best_params_, randomizedSearchAda.best_score_
gbParams = {'loss' : ['deviance', 'exponential'],
            'n_estimators': [10,50,100,200,400],
            'max_depth': randint(1,5),
            'learning_rate':[0.1, 0.05]}

randomizedSearchGB = RandomizedSearchCV(estimator=gbMod, param_distributions=gbParams, n_iter=10,
                                   scoring='roc_auc', fit_params=None, cv=None, verbose=2).fit(X_train, Y_train)

bestGbModFitted = randomizedSearchGB.best_estimator_.fit(X_train, Y_train)

bestAdaModFitted = randomizedSearchAda.best_estimator_.fit(X_train, Y_train)

cvDictHPO = cvDictGen(functions=[bestGbModFitted, bestAdaModFitted], scr='roc_auc')

cvDictNormalize(cvDictHPO)


def plotCvRocCurve(X, y, classifier, nfolds=5):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    from scipy import interp

    cv = StratifiedKFold(y, n_folds=nfolds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])

        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CV ROC curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15, 5)

    plt.show()


def rocZeroOne(y_true, y_predicted_porba):
    from sklearn.metrics import roc_curve
    from scipy.spatial.distance import euclidean

    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_porba[:, 1])

    best = [0, 1]
    dist = []
    for (x, y) in zip(fpr, tpr):
        dist.append([euclidean([x, y], best)])

    bestPoint = [fpr[dist.index(min(dist))], tpr[dist.index(min(dist))]]

    bestCutOff1 = thresholds[list(fpr).index(bestPoint[0])]
    bestCutOff2 = thresholds[list(tpr).index(bestPoint[1])]

    print('\n' + 'ROC曲线最佳点位置: TPR = {:0.3f}%, FPR = {:0.3f}%'.format(bestPoint[1] * 100, bestPoint[0] * 100))
    print('\n' + '最佳截止点: {:0.4f}'.format(bestCutOff1))

    plt.plot(dist)
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance to the perfect [0,1]')
    fig = plt.gcf()
    fig.set_size_inches(15, 5)

plotCvRocCurve(X, Y, randomizedSearchGB.best_estimator_)

rocZeroOne(Y_test, randomizedSearchGB.predict_proba(X_test))

plotCvRocCurve(X, Y, randomizedSearchAda.best_estimator_)

rocZeroOne(Y_test, randomizedSearchAda.predict_proba(X_test))

# lrMod.coef_

#####LR模型


data = pd.read_csv('Woetrain.csv')
#应变量
data_Y = data['SeriousDlqin2yrs']
#自变量，剔除对因变量影响不明显的变量
data_X = data.drop(['SeriousDlqin2yrs','DebtRatio','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
plotCvRocCurve(data_X, data_Y, lrMod.fit(data_X, data_Y))

Y_test = test['SeriousDlqin2yrs']
X_test = test.drop(['SeriousDlqin2yrs', 'DebtRatio', 'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
rocZeroOne(Y_test, lrMod.predict_proba(X_test))

import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
# 导入数据
data = pd.read_csv('./Woetrain.csv')
#应变量
Y = data['SeriousDlqin2yrs']
#自变量，剔除对因变量影响不明显的变量
X = data.drop(['SeriousDlqin2yrs','DebtRatio','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
# X = data.drop(['SeriousDlqin2yrs','DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
X1 = sm.add_constant(X)
logit = sm.Logit(Y, X1)
result = logit.fit()
print(result.summary())

test = pd.read_csv('./TestWoeData.csv')
Y_test = test['SeriousDlqin2yrs']
# X_test = test.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
X_test = test.drop(['SeriousDlqin2yrs', 'DebtRatio', 'NumberRealEstateLoansOrLines', 'NumberOfDependents'], axis=1)
X3 = sm.add_constant(X_test)
resu = result.predict(X3)
fpr, tpr, threshold = roc_curve(Y_test, resu)
rocauc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('真正率')
plt.xlabel('假正率')
plt.show()


lrMod.coef_

import math
# coe为逻辑回归模型的系数
coe = [-9.2342, 0.6295, 0.4615, 1.1004, 0.3725, 0.5254, 1.5906, 1.1133]

p = 20 / math.log(2)
q = 600 - 20 * math.log(20) / math.log(2)
baseScore = round(q + p * coe[0], 0)

# 求分数
def get_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores


x1 = get_score(coe[1], woex1, p)
x2 = get_score(coe[2], woex2, p)
x3 = get_score(coe[3], woex3, p)
x5 = get_score(coe[4], woex5, p)
x6 = get_score(coe[5], woex6, p)
x7 = get_score(coe[6], woex7, p)
x9 = get_score(coe[7], woex9, p)

print(x1, x2, x3, x5, x6, x7, x9)

# 分数计算
def compute_score(series, cut, score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list

test1 = pd.read_csv('TestData.csv')
test1['BaseScore'] = Series(np.zeros(len(test1))) + baseScore
test1['x1'] = Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
test1['x2'] = Series(compute_score(test1['age'], cutx2, x2))
test1['x3'] = Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
test1['x5'] = Series(compute_score(test1['MonthlyIncome'], cutx5, x5))
test1['x6'] = Series(compute_score(test1['NumberOfOpenCreditLinesAndLoans'], cutx6, x6))
test1['x7'] = Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7))
test1['x9'] = Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))
test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] +test1['x9']  + baseScore
test1.to_csv('ScoreData.csv', index=False)

