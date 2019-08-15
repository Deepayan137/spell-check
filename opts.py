
from pprint import pprint
from warnings import warn


def base_opts(parser):
    parser.add_argument("-c", "--config", type=str,
                        required=True, help="/path/to/config")
    parser.add_argument("-t", "--type", type=str,
                        default='CRNN', help="model type")
    parser.add_argument("-d", "--depth", type=int, default=2, help="depth")
    parser.add_argument("-fr", "--fraction", type=float,
                        default=1, help="fraction/data")
    parser.add_argument("-e", "--epochs", type=float,
                        default=20, help="no/epochs")
    parser.add_argument("--test", dest='test',
                        action='store_true', help="evaluate")
    parser.add_argument("--stn", dest='stn', action='store_true', help="STN")


def test_opts(parser):
    parser.add_argument("-c", "--config", type=str,
                        required=True, help="/path/to/config")
    # parser.add_argument("-t", "--type", type=str, required=True, help="model")


class Config:
    # data
    path = '/ssd_scratch/cvit/deep/data/'
    language = 'Hindi'
    lr = 1e-3
    epochs = 20
    imgH = 32
    nHidden = 512
    split = 'train'
    save_dir = 'saves'
    save_every = 2
    percent = 0.2
    arch = 'crnn'
    book = '1501'
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                warn("Unknown option {}, adding anyway".format(k))
                #raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}
