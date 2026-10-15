'''
feature extractors for a given image
'''
import torch
import torchvision.transforms.v2 as v2
import copy

def pre_processing(x, do_crop, convert, invert, do_filter, device, h_optimus=False):
    if do_crop:
        transform = v2.Compose([v2.RandomCrop(size=(x.shape[-2]//4,x.shape[-1]//4)), 
                                v2.Resize((256,256))])
        x = transform(x)
    if invert:
        invert = v2.RandomInvert(p=1)
        x = invert(x)
    if do_filter:
        pink_purple_color = torch.tensor([1.0, 182/255.0, 193/255.0]).view(3, 1, 1).to(device)
        x = 0.9*x + 0.1*pink_purple_color
        x = torch.clamp(x, max=1, min=0)
    if convert:
        x = convert(x, device)
    else:
        if h_optimus:
            pre_prop = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.707223, 0.578729, 0.703617], std=[0.211883, 0.230117, 0.177517]),
                                v2.Resize((224,224))])
        else:
            pre_prop = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                v2.Resize((256,256))])
        x = pre_prop(x)
    centrecrop = v2.CenterCrop((224,224))
    x = centrecrop(x)
    x = torch.clamp(x, max=1, min=0) 
    return x

def train_processing(x):
    aug_transform = v2.Compose([v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        # v2.RandomAffine(degrees=0, translate=(0.1, 0.1),shear=0.1),
        ])
    x = aug_transform(x)
    return x

def owkin_features(model, device, image_processor, x,
                   convert=None, invert=False, do_filter=False, do_crop=False, train=False, 
                   return_x = False, return_attn = False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        if do_crop:
            transform = v2.Compose([v2.RandomCrop(size=(x.shape[-2]//4,x.shape[-1]//4)), 
                                    v2.Resize((256,256))])
            x = transform(x)
        if invert:
            invert = v2.RandomInvert(p=1)
            x = invert(x)
        if do_filter:
            pink_purple_color = torch.tensor([1.0, 182/255.0, 193/255.0]).view(3, 1, 1).to(device)
            x = 0.9*x + 0.1*pink_purple_color
            x = torch.clamp(x, max=1, min=0)
        if return_x:
            img_transform = copy.deepcopy(x)
        if convert:
            x = convert(x, device)
            img_convert = copy.deepcopy(x)
        else:
            todytpe = v2.ToDtype(torch.float32, scale=True)
            x = todytpe(x)
        if train:
            x = train_processing(x)
        x = torch.clamp(x, max=1, min=0) #correct for float overflow
        inputs = image_processor(x, return_tensors="pt", do_rescale=False)
        outputs = model(**inputs.to(device),output_attentions=return_attn)
        extracted = outputs.last_hidden_state[:, 0, :]
    if convert and return_x and return_attn:
        return outputs.attentions[-1], img_transform, img_convert, extracted
    elif return_x and return_attn:
        return outputs.attentions[-1], img_transform, None, extracted
    else:
        return extracted

def h_optimus_0_features(model, device, x, convert=None, invert=False, do_filter=False, 
                        do_crop=False, train=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x = pre_processing(x, do_crop, convert, invert, do_filter, device, h_optimus=True)
        if train:
            x = train_processing(x)
        x = model(x)
    return x

def uni_features(model, device, x, convert=None, invert=False, do_filter=False, 
                 do_crop=False, train=False):
    transform = v2.Compose([
        v2.Resize(224),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x = pre_processing(x, do_crop, convert, invert, do_filter, device)
        if train:
            x = train_processing(x)
        x = model(transform(x))
    return x

def vit_features(model, device, x, convert=None, invert=False, do_filter=False, 
                 do_crop=False, train=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x = pre_processing(x, do_crop, convert, invert, do_filter, device)
        if train:
            x = train_processing(x)
        x = model._process_input(x)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        extracted = x[:, 0]
    return extracted

def resnet_features(model, device, x, convert=None, invert=False, do_filter=False,
                    do_crop=False, train=False):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x = pre_processing(x, do_crop, convert, invert, do_filter, device)
        if train:
            x = train_processing(x)
        # get the features
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x