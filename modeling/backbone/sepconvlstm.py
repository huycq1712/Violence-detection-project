import torch
import torch.nn as nn


class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, padding=1, kernels_per_layer=1):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out


class SepConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel, bias) -> None:
        super(SepConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel = kernel
        self.padding = (kernel // 2, kernel//2)
        self.bias = bias

        self.sepconv = DWSConv(in_channel=self.input_dim+self.hidden_dim, 
                                out_channel=4*self.hidden_dim, 
                                kernel=self.kernel, 
                                padding=self.padding,
                                )


    def forward(self, input, current_state):
        h_current, c_current = current_state

        concat_state = torch.cat([input, h_current], dim=1)
        concat_state = self.sepconv(concat_state)

        cc_i, cc_f, cc_o, cc_g = torch.split(concat_state, self.hidden_dim, dim=1)

        i = nn.Sigmoid()(cc_i)
        f = nn.Sigmoid()(cc_f)
        o = nn.Sigmoid()(cc_o)
        g = nn.Sigmoid()(cc_g)

        c_next = f*c_current + i*g
        h_next = o*nn.Tanh()(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        h, w = image_size

        return (torch.zeros(batch_size, self.hidden_dim, h, w, device='cuda'),
                torch.zeros(batch_size, self.hidden_dim, h, w, device='cuda'))


class SepConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel, num_layers, bias, batch_first, return_all_layers=False) -> None:
        super(SepConvLSTM, self).__init__()

        if not isinstance(kernel, list):
            kernel = [kernel] * num_layers

        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(SepConvLSTMCell(cur_input_dim, self.hidden_dim[i], kernel=kernel[i], bias=True))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input, hidden_state=None):

        if not self.batch_first:
            input = input.permute(1, 0, 2, 3, 4)

        b, t, c, h, w = input.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_length = input.size(1)
        cur_input = input

        for layer_indx in range(self.num_layers):

            h, c = hidden_state[layer_indx]
            output_inner = []

            for t in range(seq_length):
                h, c = self.cell_list[layer_indx](input=cur_input[:, t, :, :],
                                                current_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            #last_state_list = last_state_list[-1]

        return layer_output_list#, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []

        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))

        return init_states


def build_sepconvlstm(cfg):
    input_dim = cfg.MODEL.SEPCONVLSTM.INPUT_DIM
    hidden_dim = cfg.MODEL.SEPCONVLSTM.HIDDEN_DIM

    sepconvlstm = SepConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel=3, num_layers=1, batch_first=False, bias=True, return_all_layers=False)

    return sepconvlstm

if __name__ == '__main__':
    """
    torch.Size([1, 128, 31, 31])
    torch.Size([1, 512, 8, 8])
    """
    test_model = SepConvLSTM(input_dim=512, hidden_dim=128, kernel=3, num_layers=1, batch_first=True, bias=True, return_all_layers=False)
    a = torch.rand(1, 15, 512, 8, 8)
    a_ = test_model(a)


    print(a_.max(dim=1)[0].shape)

    #print(a_[0][0].shape)
    #print(a_[1][0][1].shape)
