import os
import argparse
import torch
import numpy as np
import random
from load_data import load_dataset
from load_model import build_model
from train import train_model
from utils import load_pretrained_teacher
from evaluation import evaluate_classification, evaluate_retrieval

#def set_seed(seed: int = 42, deterministic: bool = True):
    
#    random.seed(seed)
#    np.random.seed(seed)

#   torch.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)

#    os.environ["PYTHONHASHSEED"] = str(seed)

#    if deterministic:
#        torch.backends.cudnn.deterministic = True
#        torch.backends.cudnn.benchmark = False

#        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#        try:
#            torch.use_deterministic_algorithms(True, warn_only=True)
#        except Exception:
#            pass
#    else:
#        torch.backends.cudnn.deterministic = False
#        torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

# dataset setting
parser.add_argument('--dataset', type=str, default='flickr-30k', choices=['mmimdb', 'vqav2', 'flickr-30k', 'ms-coco'], help='name of the dataset')
# model setting
parser.add_argument('--teacher_model_1', default='clip-ViT-B-16', choices=['clip-ViT-B-16', 'clip-ViT-L-14', 'clip-RN101'], help='name of the model')
parser.add_argument('--teacher_model_2', default='clip-ViT-L-14', choices=['clip-ViT-B-16', 'clip-ViT-L-14', 'clip-RN101'], help='name of the model')
parser.add_argument('--student_model', default='clip-ViT-B-32', choices=['clip-ViT-B-32', 'clip-RN50', 'resnet-bert', 'vit-bert'], help='name of the model')
parser.add_argument('--project_dim', type=int, default=512, choices=[64, 128, 256, 512, 1024, 2048])
# training setting
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
#
parser.add_argument('--distiller', type=str, default='amd', choices=['none', 'lmd', 'albef', 'dsmd', 'kdmcse', 'dclip', 'g2d', 'amd'], help='name of the distiller')
# experiment detail
parser.add_argument('--gpu', dest='gpu', type=str, default='2', choices=['0', '1', '2', '3','4', '5', '6', '7'])
#
parser.add_argument("--enable_amd_vis", action="store_true")
#seed
#parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
#parser.add_argument('--deterministic', action='store_true', help='use deterministic CUDA operations')
#weighting
parser.add_argument("--weighting_strategy", type=str, default="amd", choices=["amd", "equal", "static", "uncertainty", "dwa", "gradnorm", ],)
parser.add_argument("--static_modality_weights", type=str, default=None,)
parser.add_argument("--static_teacher_weights", type=str, default=None,)
parser.add_argument("--static_level_weights", type=str, default=None,)
parser.add_argument("--dwa_temperature", type=float, default=2.0,)
parser.add_argument("--gradnorm_alpha", type=float, default=1.5,)
parser.add_argument("--gradnorm_learning_rate", type=float, default=1e-3,)

#proxy
parser.add_argument("--modality_proxy", type=str, default="default", choices=["default", "contrastive_loss", "matching_entropy"])
parser.add_argument("--teacher_proxy", type=str, default="default", choices=["default", "infonce_loss", "ensemble_similarity"])
parser.add_argument("--level_proxy", type=str, default="discrepancy", choices=["discrepancy", "raw_loss", "grad_norm"])
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#set_seed(args.seed, deterministic=args.deterministic)
train_loader, val_loader, test_loader, num_category = load_dataset(dataset=args.dataset, batch_size=args.batch_size)

teacher_model_1 = build_model(args.teacher_model_1, num_category, args.project_dim)
teacher_model_2 = build_model(args.teacher_model_2, num_category, args.project_dim)

load_pretrained_teacher(teacher_model_1, args.teacher_model_1, args.dataset, args.project_dim)
load_pretrained_teacher(teacher_model_2, args.teacher_model_2, args.dataset, args.project_dim)
student_model = build_model(args.student_model, num_category, args.project_dim)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model = train_model(args, train_loader, val_loader, num_category, teacher_model_1, teacher_model_2, student_model)
