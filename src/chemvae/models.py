
import torch
import torch.nn as nn
import nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

class Repeat(nn.Module):
    def __init__(self, num_times):
        super(Repeat, self).__init__()

    def forward(self, input):
        assert len(input.shape) >= 2
        new_shape = [input.shape[0] , self.num_times] + input.shape[1:]
        return input.repeat(new_shape)

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__(name='encoder')

        out_channels = int(args.conv_dim_depth * args.conv_d_growth_factor)
        conv_in_length = args.max_input_length
        conv_in_channels = args.nchars
        conv_layers = [nn.Conv1d(conv_in_channels, out_channels, int(args.conv_dim_width*args.conv_w_growth_factor), name='encoder_conv0'), nn.Tanh()]

        if args.batchnorm_conv:
            conv_layers += [nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.99, name='encoder_norm0')]

        in_channels = out_channels
        for j in range(1, args.conv_depth-1):
            out_channels = int(args.conv_dim_depth * (args.conv_d_growth_factor ** j))
            conv_layers += [nn.Conv1d(in_channels, out_channels, int(args.conv_dim_width*(args.conv_w_growth_factor**j)), name='encoder_conv'+str(j))]
            if args.batchnorm_conv:
                conv_layers += [nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.99, name='encoder_norm'+str(j))]
            in_channels = out_channels

        conv_layers += [Flatten()]
        self.conv_layers = nn.Sequential(conv_layers)

        if args.middle_layer > 0:
            out_size = args.hidden_dim*(args.hg_growth_factor**(args.middle_layer-1))
            in_size = conv_in_length*out_channels

            middle = [nn.Linear(in_size, out_size, name='encoder_dense0')]
            middle += [(args.activation)()]
            if args.dropout_rate_mid > 0:
                middle += [nn.Dropout(p=args.dropout_rate_mid)]
            if args.batchnorm_mid:
                middle += [nn.BatchNorm1d(out_size, name='encoder_dense0_norm')]

            in_size = out_size
            for i in range(2, args.middle_layer+1):
                out_size = args.hidden_dim*(args.hg_growth_factor**(args.middle_layer-i))
                middle += [nn.Linear(in_size, out_size, name='encoder_dense'+str(i))]
                middle += [(args.activation)()]
                if args.dropout_rate_mid > 0:
                    middle += [nn.Dropout(p=args.dropout_rate_mid)]
                if args.batchnorm_mid:
                    middle += [nn.BatchNorm1d(out_size, name='encoder_dense'+str(i)+'_norm')]

            self.middle = nn.Sequential(middle)
        else:
            self.middle = self.conv_layers
            out_size = conv_in_length*out_channels

        self.z_mean = nn.Linear(out_size, args.hidden_dim, name='z_mean_sample')

    def forward(self, x):
        middle = self.middle(x)
        z_mean = z_mean(middle)

        return [z_mean, middle]

class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__(name='decoder')

        z = [nn.Linear(args.hidden_dim, args.hidden_dim, name='decoder_dense0'), (args.activation)()]
        if args.dropout_rate_mid > 0:
            z += [nn.Dropout(p=args.dropout_rate_mid)]
        if args.batchnorm_mid:
            z += [nn.BatchNorm1d(args.hidden_dim, name='decoder_dense0_norm')]

        in_size = args.hidden_dim
        for i in range(1, args.middle_layer):
            out_size = args.hidden_dim * (args.hg_growth_factor ** i)
            z = [nn.Linear(args.hidden_dim, out_size, name='decoder_dense'+str(i)), (args.activation)()]
            if args.dropout_rate_mid > 0:
                z += [nn.Dropout(p=args.dropout_rate_mid)]
            if args.batchnorm_mid:
                z += [nn.BatchNorm1d(out_size, name='decoder_dense'+str(i)+'_norm')]
            in_size = out_size

        z += [Repeat(args.max_input_length)]

        if args.gru_depth > 1:
            x_dec = [nn.GRU(in_size, args.recurrent_dim, num_directions=1, name='decoder_gru0'), nn.Tanh()]

            for j in range(args.gru_depth-2):
                x_dec += [nn.GRU(in_size, args.recurrent_dim, num_directions=1, name='decoder_gru'+str(j)), nn.Tanh()]
