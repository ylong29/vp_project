import numpy as np
import UE4VirtualProduction as vp
import Tracker

class UE4VirtualProduction(object):
    def __init__(self, tracker, vp_studio):
        self.tracker = tracker
        self.vp_studio = vp_studio

    def forward(self):
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
        self.vp_studio.process()
        self.tracker.forward()

    def update(self):
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array
        self.vp_studio.process()
        self.tracker.update()

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
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

