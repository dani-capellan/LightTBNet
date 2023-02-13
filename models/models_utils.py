import torch
import losses

def getModel(model_name, configs, device, in_channels=1, outputs=2):
    '''
    Function that returns a model given its input configs.
    Input:
        model_name: str
        configs: dict - with parameters & info about experiment
        in_channels: int - image channels
        outputs: int - output classes
    Output: 
        model: PyTorch model
    '''
    
    #### LightTBNet
    if (model_name=="LightTBNet_3blocks"):
        from models import LightTBNet_3blocks, CustomResBlock
        model = LightTBNet_3blocks(in_channels=in_channels, CustomResBlock=CustomResBlock, outputs=outputs)
    elif (model_name=="LightTBNet_4blocks"):
        from models import LightTBNet_4blocks, CustomResBlock
        model = LightTBNet_4blocks(in_channels=in_channels, CustomResBlock=CustomResBlock, outputs=outputs)
    elif (model_name=="LightTBNet_5blocks"):
        from models import LightTBNet_5blocks, CustomResBlock
        model = LightTBNet_5blocks(in_channels=in_channels, CustomResBlock=CustomResBlock, outputs=outputs)
    #### ResNets and torchvision models
    elif (model_name=='ResNet18'):
        from models import ResNet, ResBlock
        model = ResNet(in_channels=in_channels, resblock=ResBlock, repeat=[2, 2, 2, 2], useBottleneck=False, outputs=outputs)
    elif (model_name=='ResNet34'):
        from models import ResNet, ResBlock
        model = ResNet(in_channels=in_channels, resblock=ResBlock, repeat=[3, 4, 6, 3], useBottleneck=False, outputs=outputs)
    elif (model_name=='ResNet50'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 4, 6, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='ResNet101'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 4, 23, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='ResNet152'):
        from models import ResNet, ResBlock, ResBottleneckBlock
        model = ResNet(in_channels=in_channels, resblock=ResBottleneckBlock, repeat=[3, 8, 36, 3], useBottleneck=True, outputs=outputs)
    elif (model_name=='DenseNet121'):
        from torchvision import models
        model = models.densenet121()
        model.features.conv0 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = torch.nn.Linear(in_features=1024, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB0'):
        from torchvision import models
        model = models.efficientnet_b0()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB1'):
        from torchvision import models
        model = models.efficientnet_b1()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB2'):
        from torchvision import models
        model = models.efficientnet_b2()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1408, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB3'):
        from torchvision import models
        model = models.efficientnet_b3()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB4'):
        from torchvision import models
        model = models.efficientnet_b4()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1792, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB5'):
        from torchvision import models
        model = models.efficientnet_b5()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2048, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB6'):
        from torchvision import models
        model = models.efficientnet_b6()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 56, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2304, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetB7'):
        from torchvision import models
        model = models.efficientnet_b7()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=2560, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_s'):
        from torchvision import models
        model = models.efficientnet_v2_s()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_m'):
        from torchvision import models
        model = models.efficientnet_v2_m()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='EfficientNetv2_l'):
        from torchvision import models
        model = models.efficientnet_v2_l()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    elif (model_name=='MobileNetv3_small'):
        from torchvision import models
        model = models.mobilenet_v3_small()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=outputs, bias=True)
    elif (model_name=='MobileNetv3_large'):
        from torchvision import models
        model = models.mobilenet_v3_large()
        model.features[0][0] = torch.nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=outputs, bias=True)
    else:
        model = None

    # # Pretrained - load weights
    # if(configs['pretrained']['enabled']):
    #     if(configs['pretrained']['model_ckpt']['path'] in ["", " "]):
    #         ckpt_path = os.path.join("./results",configs['pretrained']['model_ckpt']['experiment_name'],
    #                                  model_name,
    #                                  configs['pretrained']['model_ckpt']['optimizer_name'],
    #                                  configs['pretrained']['model_ckpt']['lossFn_name'],
    #                                  f"fold_{str(configs['pretrained']['model_ckpt']['fold'][model_name])}", "model_best.pth")
    #     else:
    #         ckpt_path = configs['pretrained']['model_ckpt']['path']
    #     checkpoint = torch.load(ckpt_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
        
    model.to(device)

    return model


def getOptimizer(optimizer_name, model, configs):
    '''
    Function that returns an optimizer given its input configs.
    Input:
        optimizer_name: str
        model: PyTorch model
        configs: dict - with parameters & info about experiment
    Output: 
        optimizer: PyTorch optimizer
    '''
    
    if (optimizer_name=="Adam"):
        optimizer = torch.optim.Adam(
            params = model.parameters(),
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'],
            betas = (configs['experimentEnv']['optim_args'][optimizer_name]['beta_1'], configs['experimentEnv']['optim_args'][optimizer_name]['beta_2']),
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
        
    elif (optimizer_name=="SGD"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="SGD_momentum"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            momentum=0.9,
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="Nesterov"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    elif (optimizer_name=="Nesterov_momentum"):
        optimizer = torch.optim.SGD(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            momentum=0.9,
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay'],
            nesterov = True
        )
    elif (optimizer_name=="RMSprop"):
        optimizer = torch.optim.RMSprop(
            params = model.parameters(), 
            lr = configs['experimentEnv']['optim_args'][optimizer_name]['learning_rate'], 
            weight_decay = configs['experimentEnv']['optim_args'][optimizer_name]['weight_decay']
        )
    else:
        optimizer = None
        
    return optimizer

def getLossFn(lossFn_name):
    '''
    Function that returns a loss function given its input configs.
    Inputs:
        lossFn_name: str
    Outputs:
        lossFn: Loss function
    '''
    if(lossFn_name=='CrossEntropy'):
        lossFn = torch.nn.CrossEntropyLoss()
    elif(lossFn_name=='FocalLoss'):
        lossFn = losses.FocalLoss()
    return lossFn