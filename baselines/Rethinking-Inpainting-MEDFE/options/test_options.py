from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='paris_256', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='20', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        parser.add_argument('--in_dir', type=str, default='./input', help='path to mask, we use the datasetsets of partial conv hear')
        parser.add_argument('--out_dir', type=str, default='./output', help='path to mask, we use the datasetsets of partial conv hear')
        parser.add_argument('--checkpoint_folder', type=str, default='./checkpoint_folder', help='path to mask, we use the datasetsets of partial conv hear')
        self.isTrain = False

        return parser