import pandas as pd
import datetime
import numpy as np


def give_serial_data(data1, serial_len=3):
    new_level_values = [0, 1, 2]
    multi_index = data1.index

    serial_index_values = [i for i in range(serial_len)]
    tuple_list = multi_index.tolist()
    new_tup_list = []
    for tup in tuple_list:
        new_tup_list.append((tup[0], tup[1], 0))
        new_tup_list.append((tup[0], tup[1], 1))
        new_tup_list.append((tup[0], tup[1], 2))
    multi_index = pd.MultiIndex.from_tuples(new_tup_list, names=['date', 'symbol', 'serial'])

    data2 = pd.DataFrame(index=multi_index, columns=data1.columns)

    date_index = multi_index.get_level_values('date').unique()
    symbol_index = multi_index.get_level_values('symbol').unique()
    # for serial_num in serial_index_values:
    #     for i in range(serial_len - 1, len(date_index)):
    #         for j in range(0, len(symbol_index)):
    #             try:
    #                 data2.loc[(date_index[i], symbol_index[j], serial_num), :] = data1.loc[
    #                                                                          (date_index[i - serial_len + serial_num + 1],
    #                                                                           symbol_index[j]), :]
    #             except KeyError:
    #                 continue
    # 遍历data2的每一项
    for i in range(len(data2)):
        the_date = data2.index[i][0]
        the_symbol = data2.index[i][1]
        the_serial = data2.index[i][2]
        # 在data1
        res = (the_serial + 1) % serial_len
        # date_index =
        # data2.iloc[i, :] = data1.loc[(the_date - res * delta, the_symbol), :]

        data2.iloc[i, :] = data1.loc[(data2.index[i][0] - datetime.timedelta(days=serial_len - 1), data2.index[i][1]), :]
    return data2


# 构造一个日期index
if __name__ == '__main__':
    date_index = pd.date_range(start='2020-01-01', end='2020-12-31', freq='M')
    # 构造一个symbol index
    symbol_index = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    # 将两个index组成multiIndex
    multi_index = pd.MultiIndex.from_product([date_index, symbol_index], names=['date', 'symbol'])
    #
    data1 = pd.DataFrame(index=multi_index, columns=['num1', 'num2'])

    # 将data1填充随机数
    data1['num1'] = np.random.randn(len(data1))
    data1['num2'] = np.random.randn(len(data1))

    data2 = give_serial_data(data1)

    print(data2)





