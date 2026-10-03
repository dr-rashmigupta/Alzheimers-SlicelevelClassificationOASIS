"""ResNet50 comparison: last ten backbone layers trainable, regularized head.
Example: python sota.py --data-dir /path/to/Data --output-dir runs/resnet50
Requires mri_common.py and keras_training.py beside this file.
"""
from mri_common import parser


def main(argv=None):
    p = parser('resnet50', 64)
    p.add_argument('--mixed-precision', action='store_true')
    args = p.parse_args(argv)
    from keras_training import run
    run(args, 'resnet50')


if __name__ == '__main__':
    main()
