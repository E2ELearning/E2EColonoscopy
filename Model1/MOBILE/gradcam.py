from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import gen_model
from opt import parse_args

args = parse_args()
model =  gen_model(args)

target_layers = [model.layer4[-1]]

input_tensor =

cam = GradCAM(model=model, target_layers=target_layers)

