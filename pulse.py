"""
功能：血液动力学参数检测
方法：传统计算
"""
import json
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy import interpolate
from scipy.fftpack import fft
from torch.serialization import save

__ALL__ = ["Hemodynamics"]

# logger = logging.getLogger()

class hemodynamics():
    def get(self, src_path, dst_path=None):
        """
        function:
            单个文件血液动力学参数检测，支持单通道和6通道数据
        parameters:
            src_path: 脉诊文件路径（txt）
            dst_path: 结果保存路径，若为空，则不保存（保存成json文件）
        return:
            res: 结果数据，返回字典
        """
        try:
            self.src_path = src_path
            self.dst_path = dst_path
            save_intermediate_res = False if self.dst_path is None else True
            self.init()
            assert self.data_id in ['beijing', 'shanghai'], 'Unidentified Data Type!'
            raw_data = self.read_data()
            if save_intermediate_res:
                self.plot(raw_data, 'raw_data.png')
            cutoff = int(len(raw_data) / 1000) * 1000
        except Exception as e:
            # logger.error("An error occured while reading pulse data: ", e)
            return {}
        
        try:
            if self.data_id=='beijing':
                norm_data = self.normalization(raw_data)
                avg_period_chs, period_feature_chs = [], []
                hemo_feature_chs = []
                processed_data = []
                for ch in norm_data.transpose():
                    denoised_data = self.__denoise(ch[:cutoff])
                    drift_removed_data = self.__remove_drift(denoised_data)
                    processed_data.append(drift_removed_data)
                    period_set = self.get_periods(drift_removed_data)
                    avg_period = self.get_average_period(period_set)
                    period_feature = self.cal_period_feature(avg_period)
                    hemo_feature = self.cal_hemo_feature(avg_period)

                    avg_period_chs.append(list(avg_period))
                    period_feature_chs.append(period_feature)
                    hemo_feature_chs.append(hemo_feature)

                processed_data = np.asarray(processed_data).transpose()
                res = {'avg_period': avg_period_chs,
                    'period_feature': period_feature_chs,
                    'hemo_feature': hemo_feature_chs}

            elif self.data_id=='shanghai':
                norm_data = self.normalization(raw_data)
                denoised_data = self.__denoise(norm_data[:cutoff])
                drift_removed_data = self.__remove_drift(denoised_data)
                processed_data = drift_removed_data
                period_set = self.get_periods(drift_removed_data)
                avg_period = self.get_average_period(period_set)
                period_feature = self.cal_period_feature(avg_period)
                hemo_feature = self.cal_hemo_feature(avg_period)
                res = {'avg_period' : list(avg_period),
                    'period_feature' : period_feature,
                    'hemo_feaure': hemo_feature}

            if save_intermediate_res:
                self.plot(processed_data, 'processed_data.png')
                self.plot(np.array(res['avg_period']), 'average_period.png')
                with open(os.path.join(self.dst_path, 'hemo.json'), 'w') as f:
                    json.dump(res, f)
            return res, processed_data
        
        except Exception as e:
            # logger.error("An error occured while extracting pulse features: ", e)
            return {}


    def init(self):
        # 判断数据类别
        # 北京数据6个通道，上海数据1个通道
        # 北京数据文件前4或6行、上海数据前1或2行是无关信息
        with open(self.src_path, 'rb') as f:
            lines = f.readlines()
            n = len(lines[-1].split())
            if n==6:
                self.data_id = 'beijing'
            elif n==1:
                self.data_id = 'shanghai'
            else:
                self.data_id = 'unknown'


    def read_data(self):
        data = []
        with open(self.src_path, 'rb') as f:
            lines = f.readlines()
            if self.data_id == 'beijing':
                for line in lines[6:]:
                    ch = [float(i) for i in line.split()]
                    data.append(ch)
            else:
                for line in lines[2:]:
                    data.append(float(line))
        return np.array(data)
        
    def normalization(self, data):
        '''
        func:
            对数据做归一化处理
        param:
            data: 从文件中读得的数据
        return:
            norm_data: 归一化的数据
        '''
        _range = np.max(data, axis=0) - np.min(data, axis=0)
        norm_data = (data - np.min(data, axis=0)) / _range 
        return norm_data


    def plot(self, data, save_name):
        if self.data_id == 'beijing':
            fig, axes = plt.subplots(2, 3, figsize=(13, 6))
            _max, _min = self.__get_extreme(data)
            for ch, ax in zip(data.transpose(), axes.flatten()):
                ax.plot(ch)
                ax.set(ylim=(_min, _max))
            label = ['left cun', 'left guan', 'left chi',
                     'right cun', 'right guan', 'right chi']
            for text, ax in zip(label, axes.flatten()):
                ax.set_title(text)
        elif self.data_id=='shanghai':
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data)

        plt.savefig(os.path.join(self.dst_path, save_name))


    def __get_extreme(self, data):
        _max, _min = [], []
        for ch in data:
            _max.append(max(ch)) 
            _min.append(min(ch))
        return max(_max)+0.2, min(_min)-0.2

        
    def __denoise(self, ch):
        ''' 
        func:
            将数据切分为长为1000的数据片，逐片滤波，最后拼接数据
            注 ： 送入的数据长度应当是1000的整数倍
            可以将1000修改为其他数值，但建议至少包含一个周期的脉搏信号
        param:
            ch: 单个通道的脉搏数据
        return:
            denoised_ch: 去除噪声之后的单通道数据
        '''
        slices = np.array(ch)
        length = len(ch)
        assert length%1000==0, 'Please read the comment of func __denoise()'
        slices = slices.reshape(-1, 1000)
        denoised_ch = []
        # 分片降噪，因为如果送进去整个序列的话效果很差
        for ch_slice in slices:
            wavelet = 'dmey'  # 'db9'
            w = pywt.Wavelet(wavelet)  # 选用Daubechies8小波
            # 计算出分解的最大useful level
            maxlev = pywt.dwt_max_level(len(ch_slice), w.dec_len)
            # Decompose into wavelet components, to the level selected
            coeffs = pywt.wavedec(ch_slice, w, level=maxlev)  # 将信号进行小波分解

            # ============= 滤波method1 =============
            # 如果噪声远小于信号本身，那么可以考虑使用下边的阈值threshold来滤波
            # 否则使用下边的滤波method2
            # 大致思路 ： 将噪声滤波, 其实就是减小 cD 系数项
            threshold = 0.3 # Threshold for filtering
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

            # ============= 滤波method2 =============
            # 如果滤波效果较差，可以将32~34取消注释
            # 思路 ： 将cd1 cd2 cd3直接置零，改善降噪效果
            # coeffs[maxlev] = np.zeros_like(coeffs[maxlev])
            # coeffs[maxlev - 1] = np.zeros_like(coeffs[maxlev - 1])
            # coeffs[maxlev - 2] = np.zeros_like(coeffs[maxlev - 2])
            # 信号重构
            denoised_ch.append(list(pywt.waverec(coeffs, w)))
        denoised_ch = np.array(denoised_ch)
        denoised_ch = denoised_ch.reshape(length,)
        return denoised_ch


    def __remove_drift(self, data, show_keypoints=False, show_contrast=False):
        '''
        func:
            对去除噪声之后的数据处理，去除其中的基准漂移
        param:
            data: 降噪处理后的数据, 是单通道数据
            show_keypoints: 绘制去除漂移处理过程中识别到的关键点
            show_contrast: 绘制去除漂移前后的信号图
        return:
            data: 去除漂移之后的数据
        others:
            这里注明了函数内部一些重要变量的意义
            fs: 采样频率
            Tn: fft后估算的脉搏频率，用于计算可能的周期个数
            custom_step : 经验值设定的步长，用于搜索peak
            ti : 第i个时间点
        '''
        fft_result = fft(data)
        yf = abs(fft_result) / (0.5 * len(fft_result))
        # fs是重要参数，脉诊采集设备的采样频率
        # 如果不确定fs大小，指定为1000通常是合适的，如果有问题可以按照100的幅度上下调整
        fs = 1000
        # 获取幅度最大的点对应的频率，认为是信号的主频率
        # fft的结果与原始数据的长度一致，但是因为脉诊数据的频率较低
        # 因为fft结果是对称的，所以只需在前int(length/2)个数据中选取最大值
        # 注意argmax取值范围是yf[1~int(length/2)]，因为yf[0]对应的是信号的偏移量
        # fn用于估计单个脉搏信号的频率
        length = len(data)
        start = 0
        fn = 0
        while fn == 0:
            start += 1
            fn = np.argmax(yf[start:int(length/2)]) / len(yf[1:]) * fs
        try:
            Tn = int(1 / fn * fs) # BUG: 北中医数据出现除0错误
            if Tn > 1200:
                Tn = 1000 if self.data_id=='beijing' else 800
            elif Tn < 200:
                Tn = 200
        except Exception as e:
            # 这个800其实是经过fft推算的，，，，，
            # 但是有的数据fft的结果不合理，
            # 这种情况下使用固定的值
            Tn = 800
        # 使用上边的周期计算采样时间内的周期个数
        n = len(data) // Tn
        # 从数据头选取第一个周期的峰值点，作为开始
        peak_pos = []
        t1 = np.argmax(data[:1000])
        peak_pos.append(t1)
        # 自定义搜索步长的基准值
        if Tn<1000 and Tn>300:
            custom_step = 700 if self.data_id=='beijing' else Tn
        elif Tn<300:
            custom_step = 700 if self.data_id=='beijing' else 300
        else:
            custom_step = Tn
        for i in range(n):
            start = int(peak_pos[i] + 0.5 * custom_step)
            end = int(peak_pos[i] + 1.5 * custom_step)
            # 留下的数据不足一个周期就不再检测峰值
            if len(data) - start < Tn:
                break
            try:
                ti = np.argmax(data[start: end])
                peak_pos.append(ti + start)
            except Exception as e:
                # print(e)
                break
        # 检查peak_pos最后一个元素与数据总长度的差异
        # 如果还有较长一段数据，那就继续寻找peak
        if len(data) - peak_pos[-1] > Tn:
            cnt = (len(data) - peak_pos[-1]) // Tn
            offset = len(peak_pos)
            for i in range(offset - 1, cnt + offset - 1):
                start = int(peak_pos[i] + 0.5*custom_step)
                end = int(peak_pos[i] + 1.5*custom_step)
                try:
                    ti = np.argmax(data[start: end])
                    peak_pos.append(ti + start)
                except Exception as e:
                    # print(e)
                    break
        # 检查peak_pos第一个元素与起始位置距离
        # 如果还有较长一段数据，那就继续寻找peak
        if peak_pos[0] > 0.5 * Tn:
            end = int(peak_pos[0] - 0.2 * Tn)
            pos = np.array(np.argmax(data[:end])).reshape(1,)
            peak_pos = np.concatenate((pos, peak_pos))
        # 根据peak定位波谷
        # bingo~
        if peak_pos[0] > 0.1 * Tn:
            valley_pos = [np.argmin(data[0: peak_pos[0]])]
        else:  # 第一个峰值点距离起点太近
            valley_pos = []
        offset = len(valley_pos)
        for i, pos in enumerate(peak_pos[:-1]):
            valley_pos.append(np.argmin(data[pos: peak_pos[i + 1]]) + pos)
        # 检查valley_pos最后一个元素与数据总长度的差异
        # 如果还有较长一段数据，那就继续寻找valley
        if len(data) - valley_pos[-1] > 0.5 * Tn:
            valley_pos.append(np.argmin(data[peak_pos[-1]:]) + peak_pos[-1])
        # 读取波谷振幅
        valley = data[valley_pos]
        # 检查valley_pos第一个元素与起始位置距离
        # 如果还有较长一段数据，那就继续寻找valley
        if valley_pos[0] > 0.5 * Tn:
            start_peak = np.argmax(data[:valley_pos[0]])
            if start_peak > 0.2 * Tn:
                pos = np.array(np.argmin(data[:start_peak])).reshape(1,)
                valley_pos = np.concatenate((pos, valley_pos))
                value = np.array(data[valley_pos[0]]).reshape(1,)
            else:
                pos = np.array(0).reshape(1,)
                valley_pos = np.concatenate((pos, valley_pos))
                average = np.mean(valley)
                value = np.array(average).reshape(1,)
            valley = np.concatenate((value, valley))
        # 波谷检验，剔除异常值
        mean = np.mean(valley)
        del_idx = []  # 待删除元素的索引
        for i, (idx, value) in enumerate(zip(valley_pos[:-1], valley[:-1])):
            if abs(value) < 10 * abs(mean): 
                pass
            else:
                delta_t1 = idx - valley_pos[i - 1]
                delta_t2 = valley_pos[i + 1] - idx
                if delta_t1 + delta_t2 < 1.5 * Tn:
                    del_idx.append(i)
        valley_pos = np.array(valley_pos)
        valley = np.array(valley)
        valley_pos = np.delete(valley_pos, del_idx)
        valley = np.delete(valley, del_idx)
        # 插值
        drift = np.arange(0, len(data), 1)
        # 三次插值
        f_linear = interpolate.CubicSpline(valley_pos, valley, extrapolate=True)
        # 绘图
        if show_keypoints:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data)
            ax.scatter(peak_pos, data[peak_pos], c='r')
            ax.scatter(valley_pos, valley, c='orange')
            ax.plot(drift, f_linear(drift))
            # plt.show()
            plt.savefig(os.path.join(self.dst_path, 'test1.png'))
        if show_contrast:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data, label=u'denoised pulse')
            ax.plot(drift, data - f_linear(drift),
                    label=u'denoised & remove drift')
            ax.legend(loc='upper right')
            # plt.show()
            plt.savefig(os.path.join(self.dst_path, 'test2.png'))
        data = data - f_linear(drift)
        return data
    
    def __period_seg(self, data, show_onsets=False):
        '''
        func:
            对去除漂移之后的数据处理，提取出其中的关键周期
            本函数的工作原理与__remote_drift()一致
        param:
            data: 降噪处理后的数据, 是单通道数据
            show_onsets: 绘制检测到的周期起点
        return:
            data: 去除漂移之后的数据
        others:
            这里注明了函数内部一些重要变量的意义
            fs: 采样频率
            Tn: fft后估算的脉搏频率，用于计算可能的周期个数
            custom_step : 经验值设定的步长，用于搜索peak
            ti : 第i个时间点
        '''
        # 跳过开头的异常值
        data = data[-40000:]
        yy = fft(data)
        yf = abs(yy) / (0.5 * len(yy))
        # stem_plot(yf)
        fs = 1000
        # 获取幅度最大的点对应的频率，认为是信号的主频率
        fn = np.argmax(yf[1:20000]) / len(yf[1:]) * fs
        try:
            Tn = int(1 / fn * 1000)
            if Tn > 1200:
                Tn = 1000
            elif Tn < 200:
                Tn = 200
        except Exception as e:
            # 这个800其实是经过fft推算的，，，，，
            # 但是有的数据fft的结果不合理，
            # 这种情况下使用固定的值
            Tn = 800
        # 使用上边的周期计算采样时间内的周期个数
        n = len(data) // Tn
        # 从数据头选取第一个周期的峰值点，作为开始
        peak_pos = []
        t1 = np.argmax(data[:1000])
        peak_pos.append(t1)
        # 自定义搜索步长的基准值
        if Tn < 1000 and Tn > 300:
            custom_step = 700 if self.data_id == 'beijing' else Tn
        elif Tn < 300:
            custom_step = 700 if self.data_id == 'beijing' else 300
        else:
            custom_step = Tn
        for i in range(n):
            start = int(peak_pos[i] + 0.5 * custom_step)
            end = int(peak_pos[i] + 1.5 * custom_step)
            # 留下的数据不足一个周期就不再检测峰值
            if len(data) - start < Tn:
                break
            try:
                ti = np.argmax(data[start: end])
                peak_pos.append(ti + start)
            except Exception as e:
                # print(e)
                break
        # 检查peak_pos最后一个元素与数据总长度的差异
        # 如果还有较长一段数据，那就继续寻找peak
        if len(data) - peak_pos[-1] > Tn:
            cnt = (len(data) - peak_pos[-1]) // Tn
            offset = len(peak_pos)
            for i in range(offset - 1, cnt + offset - 1):
                start = int(peak_pos[i] + 0.5*custom_step)
                end = int(peak_pos[i] + 1.5*custom_step)
                try:
                    ti = np.argmax(data[start: end])
                    peak_pos.append(ti + start)
                except Exception as e:
                    # print(e)
                    break
        # 检查peak_pos第一个元素与起始位置距离
        # 如果还有较长一段数据，那就继续寻找peak
        if peak_pos[0] > 0.5 * Tn:
            end = int(peak_pos[0] - 0.2 * Tn)
            pos = np.array(np.argmax(data[:end])).reshape(1,)
            peak_pos = np.concatenate((pos, peak_pos))

        # 根据peak定位波谷
        # bingo~
        if peak_pos[0] > 0.1 * Tn:
            valley_pos = [np.argmin(data[0: peak_pos[0]])]
        else:  # 第一个峰值点距离起点太近
            valley_pos = []
        offset = len(valley_pos)
        for i, pos in enumerate(peak_pos[:-1]):
            valley_pos.append(np.argmin(data[pos: peak_pos[i + 1]]) + pos)
        # 检查valley_pos最后一个元素与数据总长度的差异
        # 如果还有较长一段数据，那就继续寻找valley
        if len(data) - valley_pos[-1] > 0.5 * Tn:
            valley_pos.append(np.argmin(data[peak_pos[-1]:]) + peak_pos[-1])
        # 读取波谷振幅
        valley = data[valley_pos]
        # 检查valley_pos第一个元素与起始位置距离
        # 如果还有较长一段数据，那就继续寻找valley
        if valley_pos[0] > 0.5 * Tn:
            start_peak = np.argmax(data[:valley_pos[0]])

            if start_peak > 0.2 * Tn:
                pos = np.array(np.argmin(data[:start_peak])).reshape(1,)
                valley_pos = np.concatenate((pos, valley_pos))
                value = np.array(data[valley_pos[0]]).reshape(1,)
            else:
                pos = np.array(0).reshape(1,)
                valley_pos = np.concatenate((pos, valley_pos))
                average = np.mean(valley)
                value = np.array(average).reshape(1,)
            valley = np.concatenate((value, valley))

        pulse_valley_value = data[valley_pos]  # valley_pos处脉诊信号的实际值
        # 波谷检验，剔除异常值
        mean = np.mean(valley)
        del_idx = []  # 待删除元素的索引
        for i, (idx, value) in enumerate(zip(valley_pos[:-1], pulse_valley_value[:-1])):
            if abs(value) > 10 * abs(mean):  # ================= 这里修改为abs 0611
                del_idx.append(i)
        valley_pos = np.array(valley_pos)
        valley = np.array(valley)
        valley_pos = np.delete(valley_pos, del_idx)
        valley = np.delete(valley, del_idx)

        #绘图
        if show_onsets:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data)
            ax.scatter(peak_pos, data[peak_pos], c='y')
            ax.scatter(valley_pos, valley, c='r')
            # plt.show()
            plt.savefig(os.path.join(self.dst_path, 'test3.png'))
        return valley_pos

    def get_periods(self, data, plot=False):
        '''
        func:
            从预处理之后的数据中提取出完整的周期
        param:
            data: 降噪去除漂移处理后的数据, 是单通道数据
        return:
            pulse_set: 从中提取到的单个周期数据组成的列表
        '''
        onsets = self.__period_seg(data, plot)
        pulse_set = []
        length_set = []
        peak_set = []
        pulse_df = pd.DataFrame({})
        peak_value = 0
        max_length = 0
        for idx, onset in enumerate(onsets[:-2]):
            start = onset
            end = onsets[idx+1]
            period = data[start: end]
            pulse_set.append(period)
            peak = max(period)
            length = len(period)
            peak_set.append(peak)
            length_set.append(length)

        pulse_df['data'] = pulse_set
        pulse_df['peak'] = peak_set
        pulse_df['length'] = length_set
        sigma_coef = 0.8
        bool_id = (((pulse_df['length'].mean() - sigma_coef * pulse_df['length'].std()) <= pulse_df['length']) & 
                    ((pulse_df['length'].mean() + sigma_coef * pulse_df['length'].std()) >= pulse_df['length']))

        pulse_set = np.array(pulse_df[bool_id]['data'])
        max_value = np.array(pulse_df[bool_id]['peak'])
        max_length = np.array(pulse_df[bool_id]['length']) 

        # 剔除峰值位置太偏的数据
        peak_pos = []
        del_idx = []
        for pulse in pulse_set:
            maxpos = np.argmax(pulse)
            if maxpos < 250:    # 临时fix
                peak_pos.append(maxpos)
            # peak_pos.append(maxpos)
        mean = np.mean(peak_pos)
        for idx, pos in enumerate(peak_pos):
            if abs(pos - mean) > 50: # 考虑在这里放宽限制，避免全部剔除，原始值为20
                del_idx.append(idx)
        pulse_set = np.delete(pulse_set, del_idx)
        return pulse_set # BUG: 会出现全部剔除后返回空值的情况

    def get_average_period(self, periods):
        '''
        func:
            计算出多个周期的平均周期
        param:
            periods: 提取得到的多个周期数据列表
        return:
            avg_period: 平均周期
        '''
        max_len = max([len(period) for period in periods])
        padded_periods = []
        for period in periods:
            pad_length = max_len - len(period)
            # padded_periods.append(period+[0]*pad_length)
            padded_periods.append(np.pad(period, (0, pad_length), 'constant', constant_values=(0, 0)))
        sum_period = np.sum(np.array(padded_periods), axis=0)
        avg_period = sum_period / len(periods)
        return avg_period

    @staticmethod
    def cal_PCA_feature(train_data, test_data, n_components=20):
        '''
        func:
            计算PCA特征
        param:
            train_data: 训练集数据，可以是所有的平均周期, 数据维度: (batch_size, channels, feature_dim)
            train_data: 测试集数据，同train_data
            n_components: 主成分维度个数
        return:
            train_pca: 训练集数据PCA压缩后得到的特征
            test_pca: 测试集数据PCA压缩后得到的特征
        '''
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        train_pca = np.zeros((train_data.shape[0], train_data.shape[1], n_components))
        print(train_pca.shape)
        test_pca = np.zeros((test_data.shape[0], test_data.shape[1], n_components))
        for ch in range(train_data.shape[1]):
            train_pca[:, ch, :] = pca.fit_transform(train_data[:, ch, :])
            test_pca[:, ch, :] = pca.transform(test_data[:, ch, :])
        # pca.fit(train_data)
        # train_pca = pca.transform(train_data)
        # test_pca = pca.transform(test_data)
        return train_pca, test_pca


    def cal_period_feature(self, period):
        '''
        func:
            计算出周期特征，特征的定义来源于论文
            <Wrist blood flow signal-based computerized pulse diagnosis using spatial and spectrum features>
        param:
            period: 平均周期
        return:
            feature: 周期特征
        '''
        step = 200
        data = period
        max_pos = np.argmax(data)
        start = max_pos + int(0.5*step)
        end = max_pos + int(1.5 * step)
        peak2 = np.argmax(data[start: end]) + start
        delta = 3
        threshold = 0.04
        # 取peak2前边5的点的均值，判断peak2是否是正确的位置
        mean = np.mean(data[peak2-delta: peak2])
        while (mean - data[peak2]) > threshold:
            start = start + delta
            end = end + delta
            peak2 = np.argmax(data[start: end]) + start
            mean = np.mean(data[peak2-delta: peak2])
        valley = np.argmin(data[max_pos: peak2]) + max_pos
        # 计算空间特征
        peak1, valley, peak2 = max_pos, valley, peak2
        T = len(period)
        # 关键点之间的时间差
        delta_t1 = peak1
        delta_t2 = valley - peak1
        delta_t3 = peak2 - valley 
        # 上升沿 下降沿时间
        delta_ascent = peak1
        delta_decent = T - peak1 
        #
        hb = period[peak1]
        hc = period[valley]
        hd = period[peak2]
        #
        feature = [delta_t1 / T, delta_t2 / T, delta_t3 / T,
                delta_decent / delta_ascent, hc / hb, hd / hb]
        return feature


    def cal_hemo_feature(self, data):
        '''
        func:
            计算血液动力学特征
        param:
            data: 平均周期
        return:
            re: 血液动力学特征
        '''
        re = []
        # 数据归一化
        y = data / max(data)
        # 求谐波数对应的幅值
        fft_y = fft(y)  # 快速傅里叶变换
        N = y.size
        x = np.arange(N)  # 频率个数
        # half_x = x[range(int(N / 2))]  # 取一半区间
        abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
        # angle_y = np.angle(fft_y)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
        # 求第一个极小值
        mini = 1
        x = -1
        # 原先设置为4
        n_min = 4
        for i in normalization_half_y[:10]:
            if i <= mini:
                mini = i
                a = 0
            else:
                if a == 0:
                    n_min = x
                    break
            # print(n_min)
            x += 1
        # 求心率
        xl = 60/(data.size / 600)
        # 求PWV
        f_min = n_min * (xl / 60)
        c1 = 4 * f_min * 0.75
        re.append(c1)

        # 求Rf
        r = (normalization_half_y[1] - normalization_half_y[n_min])/normalization_half_y[1]
        # print(r)
        re.append(r)
        return re



if __name__ == '__main__':
    h = hemodynamics()
    h.get("/home/sharing/disk3/Datasets/TCM-Datasets/北中医冠心病症候2022/Processed/脉诊/宋爱菊-GXB195-钮.txt", "./out")