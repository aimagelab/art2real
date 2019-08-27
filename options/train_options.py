from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # arguments for the contextual loss
        parser.add_argument('--patch_size_1', type=int, default=16, help='width and height of the first patch size: must be > 0')
        parser.add_argument('--patch_size_2', type=int, default=0, help='width and height of the second patch size: must be >= 0')
        parser.add_argument('--patch_size_3', type=int, default=0, help='width and height of the third patch size: must be >= 0')
        parser.add_argument('--stride_1', type=int, default=6, help='stride of the first scale patches')
        parser.add_argument('--stride_2', type=int, default=0, help='stride of the second scale patches')
        parser.add_argument('--stride_3', type=int, default=0, help='stride of the third scale patches')
        parser.add_argument('--contextual_weight', type=float, default=0.1, help='weight of the contextual loss as member of the generator loss: if 0 the contextual loss will not be used')
        parser.add_argument('--k', type=int, default=5, help='number of nearest neighbor patches to retrieve')
        parser.add_argument('--preload_indexes', action='store_true', help='set true if you have enough memory available to speed up the training')
        parser.add_argument('--preload_mem_patches', action='store_true', help='set true if you have enough memory available to speed up the training')
        parser.add_argument('--which_mem_bank', type=str, default='./data_for_patch_retrieval', help='directory of memory banks of real patches')
        parser.add_argument('--artistic_masks_dir', type=str, default='masks_of_artistic_images_landscape', help='directory of artistic masks')

        self.isTrain = True
        return parser
