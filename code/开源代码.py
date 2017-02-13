# -*- coding: utf-8 -*-
# 导入所需的python库
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 特征工程
# 下面分别对user_info, bank_detail, browse_data, bill_detail, loan_data进行预处理

# user_info
# 读取数据集
user_info_train = pd.read_csv('D:/data/DataCastle/train/user_info_train.txt',
                                  header = None)
user_info_test = pd.read_csv('D:/data/DataCastle/test/user_info_test.txt',
                                 header = None)
# 设置字段（列）名称
col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
user_info_train.columns = col_names
user_info_test.columns = col_names
# 合并train、test
user_info = pd.concat([user_info_train, user_info_test])
# 将userid（用户id）设置为数据集的index，并删除原userid所在列
user_info.index = user_info['userid']
user_info.drop('userid',
                axis = 1,
                inplace = True)
# 查看处理后的数据集，输出前5行
print user_info.head(5)

# 下面的处理方式类似，我仅注释不同的地方
# bank_detail
bank_detail_train = pd.read_csv('D:/data/DataCastle/train/bank_detail_train.txt',
                                    header = None)
bank_detail_test = pd.read_csv('D:/data/DataCastle/test/bank_detail_test.txt',
                                    header = None)
col_names = ['userid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
bank_detail_train.columns = col_names
bank_detail_test.columns = col_names
bank_detail = pd.concat([bank_detail_train, bank_detail_test])
# 在该数据集中，一个用户对应多条记录，这里我们采用对每个用户每种交易类型取均值进行聚合
bank_detail_n = (bank_detail.loc[:, ['userid', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(['userid', 'trade_type']).mean()
# 重塑数据集，并设置字段（列）名称
bank_detail_n = bank_detail_n.unstack()
bank_detail_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']
print bank_detail_n.head(5)

# browse_history
browse_history_train = pd.read_csv('D:/data/DataCastle/train/browse_history_train.txt',
                                       header = None)
browse_history_test = pd.read_csv('D:/data/DataCastle/test/browse_history_test.txt',
                                       header = None)
col_names = ['userid', 'tm_encode_2', 'browse_data', 'browse_tag']
browse_history_train.columns = col_names
browse_history_test.columns = col_names
browse_history = pd.concat([browse_history_train, browse_history_test])
# 这里采用计算每个用户总浏览行为次数进行聚合
browse_history_count = browse_history.loc[:, ['userid', 'browse_data']].groupby(['userid']).sum()
print browse_history_count.head(5)

# bill_detail
bill_detail_train = pd.read_csv('D:/data/DataCastle/train/bill_detail_train.txt',
                                       header = None)
bill_detail_test = pd.read_csv('D:/data/DataCastle/test/bill_detail_test.txt',
                                       header = None)
col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state']
bill_detail_train.columns = col_names
bill_detail_test.columns = col_names
bill_detail = pd.concat([bill_detail_train, bill_detail_test])
bill_detail_mean = bill_detail.groupby(['userid']).mean()
bill_detail_mean.drop('bank_id',
                      axis = 1,
                      inplace = True)
print bill_detail_mean.head(5)

# loan_time
loan_time_train = pd.read_csv('D:/data/DataCastle/train/loan_time_train.txt',
                              header = None)
loan_time_test = pd.read_csv('D:/data/DataCastle/test/loan_time_test.txt',
                              header = None)
loan_time = pd.concat([loan_time_train, loan_time_test])
loan_time.columns = ['userid', 'loan_time']
loan_time.index = loan_time['userid']
loan_time.drop('userid',
               axis = 1,
               inplace = True)
print loan_time.head(5)

# 分别处理完以上数据集后，根据userid进行join，方式选择‘outer'，没有bill或者bank数据的user在对应字段上将为Na值
loan_data = user_info.join(bank_detail_n, how = 'outer')
loan_data = loan_data.join(bill_detail_mean, how = 'outer')
loan_data = loan_data.join(browse_history_count, how = 'outer')
loan_data = loan_data.join(loan_time, how = 'outer')

# 填补缺失值
loan_data = loan_data.fillna(0.0)
print loan_data.head(5)

# 构造新特征（这里仅举个小例子）
loan_data['time'] = loan_data['loan_time'] - loan_data['tm_encode_3']

# 对性别、职业等因子变量，构造其哑变量
category_col = ['sex', 'occupation', 'education', 'marriage', 'household']
def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col])
        dummy = dummy.add_prefix('{}#'.format(col))
        data.drop(col,
                  axis = 1,
                  inplace = True)
        data = data.join(dummy)
    return data
loan_data = set_dummies(loan_data, category_col)

# overdue_train，这是我们模型所要拟合的目标
target = pd.read_csv('D:/data/DataCastle/train/overdue_train.txt',
                         header = None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid',
            axis = 1,
            inplace = True)
# 构建模型
# 分开训练集、测试集
train = loan_data.iloc[0: 55596, :]
test = loan_data.iloc[55596:, :]
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state = 0)
train_y = train_y['label']
test_y = test_y['label']
# 这里用Logistic回归
lr_model = LogisticRegression(C = 1.0,
                              penalty = 'l2')
lr_model.fit(train_X, train_y)
# 给出交叉验证集的预测结果，评估准确率、召回率、F1值
pred_test = lr_model.predict(test_X)
print classification_report(test_y, pred_test)
# 输出测试集用户逾期还款概率，predict_proba会输出两个概率，取‘1’的概率
pred = lr_model.predict_proba(test)
result = pd.DataFrame(pred)
result.index = test.index
result.columns = ['0', 'probability']
result.drop('0',
            axis = 1,
            inplace = True)
print result.head(5)
# 输出结果
result.to_csv('result.csv')

非常荣幸收到datacastle的邀请参加这次分享，这里要说明下，以上只提供了部分代码（请大家见谅），帮助参加比赛的小白对建模过程有个大致的了解，但其实我们也不是什么大神，
所以以上代码仅供参考，有错误的地方还请大家指正哈。希望能借这个比赛的机会，认识更多数据分析、数据挖掘的爱好者，交流、学习到更多的干货，和大家共同进步，谢谢！

by 厦门古月哥欠