# 导入所需的库
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取csv文件，假设文件名为data.csv


label = pd.read_csv("/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/annotation.csv")

all_csv_path = '/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/all_fm.csv'


# 按照7:3的比例随机划分数据集，假设随机种子为0


f = open(all_csv_path, "a")

for nidx, nrow in label.iterrows():
    key = nrow[0]
    value = nrow
    if type(value[3]) == str:
        fm = value[3]
        line = ",".join([key,fm])
        f.write(line + "\n")

f.close()

data = pd.read_csv(all_csv_path)

train, test = train_test_split(data, test_size=0.3, random_state=0)


# 将划分后的数据集写入新的csv文件，假设文件名为train.csv和test.csv
train.to_csv("/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/train_fm.csv", index=False)
test.to_csv("/home/et23-maixj/mxj/DFER_Datasets/SIRV_final/test_fm.csv", index=False)
