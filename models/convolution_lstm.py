import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[step]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class Decoder(nn.Module):
    def  __init__(self, num_step, num_channel):
        super(Decoder, self).__init__()
        self._all_layers = []
        self.num_step = num_step
        self.num_channel = num_channel
        for i in range(self.num_step):
            name = 'conv{}'.format(i)
            conv = nn.Conv2d(self.num_channel, 3, 1, stride=1, padding=0)
            setattr(self, name, conv)
            self._all_layers.append(conv)

    def forward(self, input):
    	output = []
    	for i in range(self.num_step):
    		name = 'conv{}'.format(i)
    		y = getattr(self, name)(input[i])
    		output.append(y)
    	return output

class Encoder(nn.Module):
    def __init__(self, hidden_channels, sample_size, sample_duration):
        super(Encoder, self).__init__()
        self.convlstm = ConvLSTM(input_channels=3, hidden_channels=hidden_channels, kernel_size=3, step=sample_duration,
                        effective_step=[sample_duration-1])
################## W/o output decoder
        self.conv2 = nn.Conv2d(32, 3, 1, stride=1, padding=0)
################## With output decoder
#        self.decoder = Decoder(sample_duration, 32)
    def forward(self, x):
        b,t,c,h,w = x.size()
        x = x.permute(1,0,2,3,4)
        output_convlstm, _ = self.convlstm(x)
#        x = self.decoder(output_convlstm)
        x = self.conv2(output_convlstm[0])
        return x

		

def test():
#if __name__ == '__main__':
    # gradient check

    convlstm = ConvLSTM(input_channels=48, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[2,4]).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, 48, 64, 64)).cuda()
    target = Variable(torch.randn(1, 32, 64, 64)).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()

    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)


def test_convlstm():
    """Constructs a convlstm model.
    """
    model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=[112,112], sample_duration=4).cuda()
    input = Variable(torch.randn(20, 3, 4, 112, 112)).cuda()

    output = model(input)
    print(output.size())

def encoder(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Encoder(**kwargs)
    return model

#if __name__ == '__main__':
#    test_convlstm()