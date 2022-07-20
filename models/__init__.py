import torch

from .pmn import PrimitiveMatchingNetwork
from .classifier import SketchClassifier
from .communication import Communication
from .fgsbir.model import FGSBIR_Model

def get_model(args, n_classes, DEVICE):
    if args.model_type == 'none':
        model = None
    elif args.model_type == 'pmn':
        model = PrimitiveMatchingNetwork(
                                 temperature=args.temperature,
                                 use_pos_embed=args.use_pos_embed,
                                 use_sinusoid_embed=args.use_sinusoid_embed,
                                 transformations=args.transformations,
                                 use_proportion_scale=args.use_proportion_scale,
                                 transform_bound=args.transform_bound,
                                 Nz=args.Nz,
                                 dt_gamma=args.dt_gamma,
                                 learn_compatibility=args.learn_compatibility)
        model = model.to(DEVICE)

    elif args.model_type == 'classifier':
        model = SketchClassifier(n_classes=n_classes, use_pos_embed=args.use_pos_embed,
                                 use_sinusoid_embed=args.use_sinusoid_embed).to(DEVICE)
    elif args.model_type == 'communication':
        if args.loss_model_type == 'class':
            classifier = SketchClassifier(n_classes=n_classes, use_pos_embed=args.use_pos_embed,
                                          use_sinusoid_embed=args.use_sinusoid_embed).to(DEVICE)
        elif args.loss_model_type == 'fgsbir':
            classifier = FGSBIR_Model(backbone=args.fgsbir_backbone,
                                      ce_temp=args.temperature).to(DEVICE)
        elif args.loss_model_type == 'none':
            classifier = None
        else:
            raise NotImplementedError

        if args.loss_model_type != 'none':
            classifier.load_state_dict(torch.load(args.loss_model_ckpt))

        if args.loss_model_type == 'class':
            classifier = classifier.encoder

        model = Communication(classifier=classifier,
                              loss_type=args.loss_model_type,
                              comm_mode=args.comm_mode,
                              budget=args.budget,
        ).to(DEVICE)
    elif args.model_type == 'fgsbir':
        model = FGSBIR_Model(backbone=args.fgsbir_backbone,
                             ce_temp=args.temperature,
                             loss_type=args.loss_model_type).to(DEVICE)
    else:
        raise NotImplementedError

    return model

