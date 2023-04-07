from models import cnn, cnnlstm, cnnlstm_copy, deepvo
from models.resnet import resnet18
#from models.HR import FCN
def gen_model(args, device):
   
    assert args.model in [
        'cnn',
        'cnn_gression','MOBILEV2','Res18',
        'cnnlstm',
        'seq2seq'
    ]
    print(args.model)
    if args.model == 'cnn':
        
        # print('\n## Importing CNN model')
        # backbone = resnet18(pretrained = True)
        # model = FCN(backbone, in_channel=512, num_classes=1)
        # model = efficient.EffiNet(args)
        model = cnn.CNN(args)
    elif args.model == 'cnn_gression':
            
        print('\n## Importing CNN regressionmodel')

        model = cnn.CNNregression(args)
        
    elif args.model == 'Res18':          
        model = cnn.Res18(args)
        
    elif args.model == 'MOBILEV2':
        print("ok")     
        model = cnn.MOBILEV2(args)
        
    elif args.model == 'cnnlstm':
        print('\n## Importing CNNLSTM model')
        model = deepvo.EncoderCNN(args)
    
    elif args.model == 'seq2seq':
        print('\n## Importing Seq2Seq model')
        model = cnnlstm.CNNLSTM(args)
        
    return model.to(device)

