
from .ops import str2bool

def add_parse_args(parser):
    parser.add_argument('--seed', type=int, default=1111,
            help='random seed')
    parser.add_argument('--use_cuda', type=str2bool, default='True',
            help='use cuda (default: True)')
    parser.add_argument('--log_interval', type=int, default=100,
            help='logging interval')
    parser.add_argument('--when', nargs="+", type=int, default=[],
            help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--use_validation', type=str2bool, default="False",
            help='compute stats on validation set')

    # model args
    parser.add_argument('--max_input_length', type=int, default=10)
    parser.add_argument('--nchars', type=int, default=10)

    # CNN model args
    parser.add_argument('--conv_depth', type=int, default=10)
    parser.add_argument('--conv_dim_depth', type=int, default=10)
    parser.add_argument('--conv_d_growth_factor', type=float, default=70)
    parser.add_argument('--conv_dim_width', type=int, default=10)
    parser.add_argument('--conv_w_growth_factor', type=float, default=70)
    parser.add_argument('--batchnorm_conv', type=str2bool, default='True')

    parser.add_argument('--middle_layer', type=int, default=70)
    parser.add_argument('--hidden_dim', type=int, default=70)
    parser.add_argument('--hg_growth_factor', type=float, default=70)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout_rate_mid', type=float, default=0.1)
    parser.add_argument('--batchnorm_mid', type=str2bool, default='True')

    parser.add_argument('--gru_depth', type=int, default=70)
    parser.add_argument('--recurrent_dim', type=int, default=70)
    parser.add_argument('--do_tgru', type=str2bool, default='True')

    # optimizer args 
    parser.add_argument('--optimizer', type=str, default='adam',
            help='which optimizer')
    parser.add_argument('--num_epochs', type=int, default=8000,
            help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
            help='batch size')

    parser.add_argument('--clip', type=float, default=0.0,
            help='gradient clipping')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
            help='weight decay applied to all weights')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='initial learning rate')
