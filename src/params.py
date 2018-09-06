
def add_parse_args(parser):
    parser.add_argument('--seed', type=int, default=1111,
            help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
            help='use cuda (default: True)')
    parser.add_argument('--log_interval', type=int, default=100,
            help='logging interval')
    parser.add_argument('--when', nargs="+", type=int, default=[],
            help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    parser.add_argument('--use_validation', type=bool, default=False,
            help='compute stats on validation set')


    # Resnet model args
    parser.add_argument('--resnet_pep_emsize', type=int, default=10)
    parser.add_argument('--resnet_blosum_em_size', type=int, default=10)
    parser.add_argument('--resnet_mhc_emsize', type=int, default=70)

    parser.add_argument('--resnet_block', type=str, default='basic')

    parser.add_argument('--resnet_layers', type=int, nargs='+', default=[5])
    parser.add_argument('--resnet_conv_filters', type=int, default=64)

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
