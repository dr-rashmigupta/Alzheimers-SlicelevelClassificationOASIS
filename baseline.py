"""VGG19 baseline: frozen ImageNet backbone and 256-unit classification head.
Example: python baseline.py --data-dir /path/to/Data --output-dir runs/vgg19
Requires mri_common.py and keras_training.py beside this file.
"""
from mri_common import parser


def main(argv=None):
    p = parser('vgg19', 128)
    p.add_argument('--mixed-precision', action='store_true')
    args = p.parse_args(argv)
    from keras_training import run
    run(args, 'vgg19')


if __name__ == '__main__':
    main()
