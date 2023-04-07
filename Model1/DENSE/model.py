from models import cnn, cnnlstm
from models.resnet import resnet18
from models.HR import FCN
def gen_model(args, device):
   
    assert args.model in [
        'cnn',
        'cnnlstm',
        'dense121'
    ]
    
    if args.model == 'cnn':
        
        # print('\n## Importing CNN model')
        # backbone = resnet18(pretrained = True)
        # model = FCN(backbone, in_channel=512, num_classes=1)
        # model = efficient.EffiNet(args)
        model = cnn.CNN(args)
        
    elif args.model == 'dense121':   
        print("dense121")       
        model = cnn.DensNet(args) 
    
    elif args.model == 'cnnlstm':
        print('\n## Importing Seq2Seq model')
        model = cnnlstm.CNNLSTM(args)
        
    return model.to(device)
        