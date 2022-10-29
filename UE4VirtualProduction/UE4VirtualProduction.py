import numpy as np
import UE4VirtualProduction as vp

# 用UE4VirtualProduction类来实现UE4VirtualProduction build result SDK的封装。
class UE4VirtualProduction(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height, filter_height, zero_padding,
                stride)
        self.output_array = np.zeros((self.filter_number,
                                      self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def backward(self, output):
        return 1 if output > 0 else 0

    # forward方法实现了sdk的调用，下面是代码实现
    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array,
                                          self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        element_wise_op(self.output_array,
                        self.activator.forward)

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,
                         self.input_height, self.input_width))

    def bp_gradient(self, sensitivity_array):
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

