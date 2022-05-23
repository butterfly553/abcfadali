import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class LoadData(Dataset):  # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, data_path, num_nodes, divide,time_interval ,history_length, train_mode):
        """
        :param data_path: list, ["graph file name" , "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes.
        :param divide_days: list, [ days of train data, days of test data], list to divide the original data.
        :param time_interval: int, time interval between two traffic data records (mins).---5 mins
        :param history_length: int, length of history data to be used.
        :param train_mode: list, ["train", "test"].
        """

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_num = divide[0]  # 5073, train_data
        self.test_num = divide[1]  # 1000  ,test_data
        self.history_length = history_length  # 历史长度为4
        self.time_interval = time_interval     # 间隔为5
        self.flow_data = np.load(data_path[0]) 
        self.graph = pd.read_csv('/content/abcfadali/adj.csv').to_numpy()
    def __len__(self):  # 表示数据集的长度
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.train_num * self.time_interval#　训练的样本数　＝　训练集总长度　－　历史数据长度
        elif self.train_mode == "test":
            return self.test_num * self.time_interval #　每个样本都能测试，测试样本数　＝　测试总长度
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # 功能是如何取每一个样本 (x, y), index = [0, L1 - 1]这个是根据数据集的长度确定的
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":
            index = index #　训练集的数据是从时间０开始的，这个是每一个流量数据，要和样本（ｘ,y）区别
        elif self.train_mode == "test":
            index += self.train_num * self.time_interval #　有一个偏移量
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode) # 这个就是样本（ｘ,y）

        data_x = LoadData.to_tensor(data_x)  # [N, H, D] # 转换成张量
        data_y = LoadData.to_tensor(data_y).unsqueeze(2)  # [N, 1, D]　# 转换成张量，在时间维度上扩维

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y} #组成词典返回
    @staticmethod
    def slice_data(data, history_length, index, train_mode): # 根据历史长度,下标来划分数据样本
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index #开始下标就是时间下标本身，这个是闭区间
            end_index = index + history_length #结束下标,这个是开区间
        elif train_mode == "test":
            start_index = index - history_length #开始下标，这个最后面贴图了，可以帮助理解
            end_index = index # 结束下标
        else:
            raise ValueError("train model {} is not defined".format(train_mode))
        data_x = data[:, :,start_index:end_index]  # 在切第二维，不包括end_index
        data_y = data[:, :,end_index]  # 把上面的end_index取上

        return data_x, data_y
    @staticmethod
    def to_tensor(data):
        return torch.tensor(data)
