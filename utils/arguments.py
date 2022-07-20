import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-dir', default='./data', help='Root directory for the data')
parser.add_argument('--preprocessed-root', default=None, help='Path to preprocessed data')
parser.add_argument('--dataset', default='quickdraw09', help='Dataset to train/test')
parser.add_argument('--model-type', default='pmn', help='Model type')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--weight-decay', type=float, default=0., help='Weight decay for optimizer')
parser.add_argument('--Nz', type=int, default=256, help='Latent dimension')
parser.add_argument('--dt-gamma', type=float, default=6.,  help='Gamma for distance transform')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for contrastive loss')
parser.add_argument('--transformations', type=str, default='rsr')
parser.add_argument('--use-proportion-scale', action='store_true', default=False)
parser.add_argument('--transform-bound', type=float, default=None)
parser.add_argument('--learn-compatibility', action='store_true', default=False)
parser.add_argument('--n-log-img', type=int, default=50, help='Number of images to display in the logger')
parser.add_argument('--log-dir', default='./log', help='Root directory for the logs')
parser.add_argument('--log-every', type=int, default=500,  help='How often to log')
parser.add_argument('--save-every', type=int, default=5000,  help='How often to save model')
parser.add_argument('--log-name', default='test', help='Name of the logs')
parser.add_argument('--loss-model-ckpt', type=str, default=None, help='Checkpoint used as the loss model')
parser.add_argument('--loss-model-type', default='cross_entropy', help='Loss to use for the loss model')
parser.add_argument('--view-scale', type=int, default=256,  help='Size of log images')
parser.add_argument('--test', default=None, type=str, help='Checkpoint used for test set evaluation')

# Position embedding
parser.add_argument('--use-pos-embed', action='store_true', default=False, help='Use positional embeddings')
parser.add_argument('--use-sinusoid-embed', action='store_true', default=False, help='Use sinusoid positional embeddings')

#FGSBIR
parser.add_argument('--fgsbir-backbone', default='InceptionV3')
parser.add_argument('--load-fgsbir-photos', action='store_true', default=False, help='Load photo images for retrieval task')

#Communication
parser.add_argument('--comm-mode', default='human')
parser.add_argument('--budget', type=float, default=1.0)

# Data optimization
parser.add_argument('--no-stroke-resampling', action='store_true', default=False)
parser.add_argument('--max-stroke-length', type=int, default=25)

# Save processed images
parser.add_argument('--add-comm-model', action='store_true', default=False)
parser.add_argument('--only-save-coords', action='store_true', default=False)
