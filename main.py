import os
import datetime

import numpy as np
import torch

from models import get_model
from utils.data import get_data
from utils.arguments import parser
from trainer import Trainer

if __name__ == '__main__':
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Add timestamp to logdir
    if args.log_name is None:
        LOG_DIR = os.path.join(args.log_dir,
                               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        LOG_DIR = os.path.join(args.log_dir,
                               args.log_name)

    if args.test is not None:
        torch.manual_seed(42)
        np.random.seed(42)

    dataloader_coords, n_classes, label_names = get_data(
            args.data_dir, args.dataset, batch_size=args.batch_size,
            train=True,
            preprocessed_root=args.preprocessed_root,
            load_fgsbir_photos=args.load_fgsbir_photos,
            resample_strokes=not args.no_stroke_resampling,
            max_stroke_length=args.max_stroke_length
    )

    test_dataloader_coords, _, _ = get_data(
            args.data_dir, args.dataset, batch_size=args.n_log_img,
            train=False,
            preprocessed_root=args.preprocessed_root,
            load_fgsbir_photos=args.load_fgsbir_photos,
            resample_strokes=not args.no_stroke_resampling,
            max_stroke_length=args.max_stroke_length
    )

    model = get_model(args, n_classes, DEVICE)

    if args.test is not None and not args.model_type == 'communication':
        model.load_state_dict(torch.load(args.test))

    if args.test is not None:
        torch.manual_seed(42)
        np.random.seed(42)

    trainer = Trainer(args.model_type, model, args.test is None, args.learning_rate, args.epochs, dataloader_coords, test_dataloader_coords, label_names, DEVICE, LOG_DIR, args.log_every, args.n_log_img, args.save_every, args.view_scale, args.weight_decay)
    trainer.run()
