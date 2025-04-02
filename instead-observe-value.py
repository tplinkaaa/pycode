import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def process_data():
    file_path = r'C:\Users\97427\Desktop\寒假毕设\code\observe-value.xlsx'
    data = pd.read_excel(file_path)

    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    data = data[data['Year'].isin(years)]

    data = data[(data['Hour'] >= 11) & (data['Hour'] <= 14)]

    mean_data = data.groupby(['Year', 'Month', 'Day'])[data.columns.difference(['Year', 'Month', 'Day', 'Hour'])].mean()

    # 合并 Year、Month、Day 为新的一列 'date'
    mean_data['date'] = mean_data.index.map(lambda x: f"{x[0]}{str(x[1]).zfill(2)}{str(x[2]).zfill(2)}")
    mean_data = mean_data.reset_index(drop=True)
    # 直接从索引中提取 Year、Month、Day
    mean_data['Year'] = mean_data.index[0]
    mean_data['Month'] = mean_data.index[1]
    mean_data['Day'] = mean_data.index[2]
    mean_data = mean_data.drop(columns=['Year', 'Month', 'Day'])

    # 将 'date' 列移到第一列
    cols = mean_data.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    mean_data = mean_data[cols]

    # 生成六年的完整日期序列
    all_dates = []
    for year in years:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        all_dates.extend([start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)])
    all_dates_str = [date.strftime('%Y%m%d') for date in all_dates]

    # 通过重新索引来补充空缺日期并填充为NaN
    mean_data = mean_data.set_index('date').reindex(all_dates_str).reset_index().rename(columns={'index': 'date'})
    mean_data = mean_data.fillna(np.nan)

    save_path = r'C:\Users\97427\Desktop\寒假毕设\code\mean-observe-value.csv'
    mean_data.to_csv(save_path, index=False)

def replace_columns():
    # 读取alldata.csv文件，命名为cmaq_data
    cmaq_data = pd.read_csv('alldata.csv')
    # 读取mean-observe-value.csv文件，命名为observe_data
    observe_data = pd.read_csv('mean-observe-value.csv')

    # 获取cmaq_data中除了date列之外的列名列表
    cmaq_columns = cmaq_data.columns[1:]

    for col in cmaq_columns:
        if col in observe_data.columns:
            # 使用observe_data中的对应列替换cmaq_data中的列
            cmaq_data[col] = observe_data[col]

    # 将替换后的cmaq_data保存为alldata-instead.csv
    cmaq_data.to_csv('alldata-instead.csv', index=False)


if __name__ == "__main__":
    process_data()
    replace_columns()