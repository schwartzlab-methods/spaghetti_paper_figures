import pandas as pd
import torch
import skimage.transform as st
from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torchvision.transforms import v2
import numpy as np
from segment import build_segment_model, cellpose_he, compute_dice_score, get_mask_per_instance, stardist_he
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr
import os
import altair as alt

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")


def crop_center(img,cropx,cropy):
    '''
    Centre crop an numpy image
    '''
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

class Regression(torch.nn.Module):
    def __init__(self, input_dim):
        super(Regression, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        # x = torch.nn.functional.sigmoid(x)
        return x

class Classification(torch.nn.Module):
    def __init__(self, num_classes, input_dim):
        super(Classification, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x) # no softmax here, as CrossEntropyLoss includes it
        return x

def build_model(model, task="classification"):
    class ClassificationTask(pl.LightningModule):
        def __init__(self, model):
            super(ClassificationTask, self).__init__()
            self.model = model
            self.validation_step_outputs = []
            self.criterion = torch.nn.CrossEntropyLoss()
            # track prediction parameters
            self.true = []
            self.pred = []
            self.prob = []

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            class_labels = labels[0]
            output = self(images)
            loss = self.criterion(output, class_labels)
            self.log("train_loss", loss.item(), prog_bar=True)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters(), lr=0.001)

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            class_labels = labels[0]
            output = self(images)
            loss = self.criterion(output, class_labels)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == class_labels).sum().item()
            total = class_labels.size(0)
            f1 = f1_score(class_labels.view(-1).cpu(), predicted.view(-1).cpu(), average='weighted')
            ret = {'val_loss': loss, 'correct': correct, 'total': total, "f1": f1}
            self.validation_step_outputs.append(ret)
            return ret

        def on_validation_epoch_end(self):
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            correct = sum([x['correct'] for x in self.validation_step_outputs])
            total = sum([x['total'] for x in self.validation_step_outputs])
            f1_sum = sum([x['f1'] for x in self.validation_step_outputs])
            acc = 100*correct/total
            f1_avg = f1_sum / total
            dic = {'val_loss': avg_loss, 'val_acc': acc, "f1_score": f1_avg}
            self.log_dict(dic, sync_dist=True,prog_bar=True)
            self.validation_step_outputs=[]

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            images, labels = batch
            output = self(images)
            self.prob.append(torch.nn.functional.softmax(output, dim=1).squeeze().cpu().detach().numpy())
            _, predicted = torch.max(output.data, 1)
            self.pred.append(predicted.cpu().item())
            self.true.append(labels[0].item())
    
    class RegressionTask(pl.LightningModule):
        def __init__(self, model):
            super(RegressionTask, self).__init__()
            self.model = model
            self.validation_step_outputs = []
            self.criterion = torch.nn.MSELoss()
            # track prediction parameters
            self.true = []
            self.pred = []
            self.loss = []

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            output = self(images)
            loss = self.criterion(output, labels)
            self.log("train_loss", loss.item(), prog_bar=True)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.AdamW(self.model.parameters(), lr=0.001)

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            output = self(images)
            loss = self.criterion(output, labels)
            self.log("val_loss", loss.item(), prog_bar=True, sync_dist=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            images, labels = batch
            output = self(images)
            loss = self.criterion(output, labels)
            self.pred.append(output.cpu().item())
            self.true.append(labels.item())
            self.loss.append(loss.cpu().item())

    if task == "classification":
        task_model = ClassificationTask(model)
    elif task == "regression":
        task_model = RegressionTask(model)
    else:
        raise ValueError("Task not supported")
    return task_model

def segmentation(data_loader, convertion, model_name: list[str], do_convert: list[bool],
                 do_invert: list[bool], coco=None, masks=None, do_crop=False, utom=False):
    model_dict = {"cellpose": ["cyto3", "tissuenet_cp3"],
                  "stardist": ["2D_versatile_fluo", "2D_versatile_he"]}
    crop_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dice_L = []
    seg_result_nonbinary = []
    seg_result_binary = []
    img_L = []
    model_type = ["cellpose" if name in model_dict["cellpose"] else "stardist" for name in model_name]
    model_L = [build_segment_model(name=m_name, model_type=m_type) for m_name, m_type in zip(model_name, model_type)]
    for idx, (img, label) in enumerate(tqdm(data_loader)):
        f_path = label[2][0]
        if do_crop and not utom:
            crop = v2.CenterCrop((crop_size,crop_size))
            img = crop(img)
        # reshape to 256x256
        resize = v2.Resize((256,256))
        img = resize(img)
        # get the masks
        if coco:
            coco_masks = get_mask_per_instance(coco, f_path)
        elif masks:
            f_name = f_path.split("/")[-1]
            add = [x for x in masks if f_name in x]
            if len(add) == 0:
                print("No masks found, continue")
                continue
            coco_masks = np.array(Image.open(add[0]))
        else:
            raise ValueError("No masks provided")
        if do_crop:
            coco_masks = crop_center(coco_masks, crop_size, crop_size)
        binary_coco_masks = (coco_masks > 0).astype(np.uint8)
        # resize the masks
        binary_coco_masks = st.resize(binary_coco_masks, (256,256), order=0, preserve_range=True, anti_aliasing=False)
        coco_masks = st.resize(coco_masks, (256,256), order=0, preserve_range=True, anti_aliasing=False)
        # convert to numpy for image
        img_np = img.squeeze()
        img_np = img_np.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0)) # H,W,C
        # track some results
        temp_img_L = []
        temp_dice = []
        temp_seg_result_nonbinary = [img_np, coco_masks]
        temp_seg_result_binary = [img_np, binary_coco_masks]
        # start the segmentation
        for i, seg_model in enumerate(model_L):
            if do_invert[i]:
                invert = v2.RandomInvert(p=1)
                img = invert(img)
            if do_convert[i]:
                img_final = convertion(img, device)
            else:
                resize = v2.Resize((256,256))
                img_final = resize(img)
            # put back to numpy array
            img_final = img_final.squeeze()
            img_final = img_final.cpu().numpy()
            img_final = np.transpose(img_final, (1, 2, 0)) # to (H, W, C)   
            try:
                if model_type[i] == "cellpose":
                    pred_masks = cellpose_he(img_final, seg_model)
                else:
                    pred_masks = stardist_he(img_final, seg_model)
            except (IndexError, RuntimeError):
                print("Segmentation failed, continue")
                print(img_final.shape)
                print(np.min(img_final), np.max(img_final))
                continue
            pred_masks = np.squeeze(pred_masks)
            # make binary 
            binary_pred_masks = (pred_masks > 0).astype(np.uint8)
            assert binary_coco_masks.shape == binary_pred_masks.shape, f"Shapes of the masks do not match, got {binary_coco_masks.shape} for coco and {binary_pred_masks.shape} for segmentation"
            # compute dice
            dice = compute_dice_score(binary_pred_masks, binary_coco_masks)
            print("The dice score is: ", dice)
            # append to list
            temp_dice.append(dice)
            temp_seg_result_binary.append(binary_pred_masks)
            temp_seg_result_nonbinary.append(pred_masks)
            temp_img_L.append(img_final)
        dice_L.append(temp_dice)
        seg_result_nonbinary.append(temp_seg_result_nonbinary)
        seg_result_binary.append(temp_seg_result_binary)
        img_L.append(temp_img_L)
    return dice_L, seg_result_nonbinary, seg_result_binary, img_L

def segmentation_wang(data_loader, convertion, model_name: list[str], do_convert: list[bool],
                 do_invert: list[bool]):
    model_dict = {"cellpose": ["cyto3", "tissuenet_cp3"],
                  "stardist": ["2D_versatile_fluo", "2D_versatile_he"]}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dice_L = []
    seg_result_binary = []
    img_L = []
    model_type = ["cellpose" if name in model_dict["cellpose"] else "stardist" for name in model_name]
    model_L = [build_segment_model(name=m_name, model_type=m_type) for m_name, m_type in zip(model_name, model_type)]
    for idx, (img, label) in enumerate(tqdm(data_loader)):
        label_np = label.squeeze().cpu().numpy()
        label_np = np.transpose(label_np, (1, 2, 0))[:,:,0].reshape(256,256) # H,W only
        # convert to numpy for image
        img_np = img.squeeze()
        img_np = img_np.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0)) # H,W,C
        # track some results
        temp_img_L = []
        temp_dice = []
        temp_seg_result_binary = [img_np, label_np]
        # start the segmentation
        for i, seg_model in enumerate(model_L):
            if do_invert[i]:
                invert = v2.RandomInvert(p=1)
                img = invert(img)
            if do_convert[i]:
                img_final = convertion(img, device)
            else:
                resize = v2.Resize((256,256))
                img_final = resize(img)
            # put back to numpy array
            img_final = img_final.squeeze()
            img_final = img_final.cpu().numpy()
            img_final = np.transpose(img_final, (1, 2, 0)) # to (H, W, C)   
            try:
                if model_type[i] == "cellpose":
                    pred_masks = cellpose_he(img_final, seg_model)
                else:
                    pred_masks = stardist_he(img_final, seg_model)
            except IndexError:
                continue
            pred_masks = np.squeeze(pred_masks)
            # make binary 
            binary_pred_masks = (pred_masks > 0).astype(np.uint8)
            assert label_np.shape == binary_pred_masks.shape, f"Shapes of the masks do not match, got {label_np.shape} for coco and {binary_pred_masks.shape} for segmentation"
            # compute dice
            dice = compute_dice_score(binary_pred_masks, label_np)
            print("The dice score is: ", dice)
            # append to list
            temp_dice.append(dice)
            temp_seg_result_binary.append(binary_pred_masks)
            temp_img_L.append(img_final)
        dice_L.append(temp_dice)
        seg_result_binary.append(temp_seg_result_binary)
        img_L.append(temp_img_L)
    return dice_L, seg_result_binary, img_L


def random_forest_classifier(loader, feature_extractor, device, save_path=None, load=None):
    # data prep
    print("Preparing data...")
    if load:
        data = np.load(load)
        data_features = data['features']
        data_labels = data['labels']
    else:
        data_features = []
        data_labels = []
        resize = v2.Resize((256, 256))
        for img, label in tqdm(loader):
            img = resize(img)
            features = feature_extractor(img, device, True)
            data_features.append(features.cpu().numpy())
            data_labels.append(label[0].cpu().numpy())
        data_features = np.concatenate(data_features, axis=0)
        data_labels = np.concatenate(data_labels, axis=0)

        if save_path: # save the features and labels if a path is provided
            np.savez_compressed(save_path, features=data_features, labels=data_labels)
    # split to 10 fold
    print("Training starting ...")
    prob_all_fold = []
    label_all_fold = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in tqdm(skf.split(data_features, data_labels)):
        X_train, X_test = data_features[train_idx], data_features[test_idx]
        y_train, y_test = data_labels[train_idx], data_labels[test_idx]

        # train the classifier
        clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
        # clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
    
        # predict and return the probabiities
        pred_prob = clf.predict_proba(X_test)

        # add to list
        prob_all_fold.append(pred_prob)
        label_all_fold.append(y_test)

    return prob_all_fold, label_all_fold

def random_forest_with_increasing_training_size(loader, feature_extractor, device):
    # data prep
    print("Preparing data...")
    data_features = []
    data_labels = []
    for img, label in tqdm(loader):
        features = feature_extractor(img, device, True)
        data_features.append(features.cpu().numpy())
        data_labels.append(label[0].cpu().numpy())
    data_features = np.concatenate(data_features, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)

    # random split 80-20
    split_data = train_test_split(data_features, data_labels, test_size=0.2, random_state=42, stratify=data_labels)
    X_train, X_test, y_train, y_test = split_data
    # training the classifier, using 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100% of the training data
    prob_all_fold = []
    label_all_fold = []
    for i in range(1, 11):
        print(f"Training with {i*10}% of the training data")
        # split and ensure stratification
        test_size = 1 - i * 0.1
        if test_size == 0:
            X_train_subset, y_train_subset = X_train, y_train
        else:
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, random_state=42, 
                                                                    stratify=y_train, test_size=test_size)
        
        # train the classifier
        clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
        clf.fit(X_train_subset, y_train_subset)
    
        # predict and return the probabiities
        pred_prob = clf.predict_proba(X_test)

        # add to list
        prob_all_fold.append(pred_prob)
        label_all_fold.append(y_test)
    
    return prob_all_fold, label_all_fold, list(range(10, 101, 10))

def random_forest_with_increasing_noise(loader, feature_extractor, device):
    '''
    gradually add noises to the image to evaluate the robustness of the model
    '''
    noise = v2.GaussianBlur(kernel_size=(5,5))
    # data prep
    print("Preparing data...")
    data_features_all = {k:[] for k in range(11)} # 0 times to 10 times of noise addition
    data_images = {k:[] for k in range(11)} # save images for visualization
    data_labels = []
    for img, label in tqdm(loader):
        data_labels.append(label[0].cpu().numpy())
        # from 0 times to 10 times add noise
        for i in range(11):
            if i == 0:
                img_noisy = img
            else:
                img_noisy = noise(img_noisy)
            features = feature_extractor(img_noisy, device, True)
            data_features_all[i].append(features.cpu().numpy())
            data_images[i].append(img_noisy.cpu().squeeze().numpy())
    data_labels = np.concatenate(data_labels, axis=0)
    
    final_dict = {}
    for k in data_features_all.keys():
        print(f"training data with {k} times noise added")
        data_features = np.concatenate(data_features_all[k], axis=0)
        # split to 10 fold
        print("Training starting ...")
        prob_all_fold = []
        label_all_fold = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for train_idx, test_idx in tqdm(skf.split(data_features, data_labels)):
            X_train, X_test = data_features[train_idx], data_features[test_idx]
            y_train, y_test = data_labels[train_idx], data_labels[test_idx]

            # train the classifier
            clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
            clf.fit(X_train, y_train)
        
            # predict and return the probabiities
            pred_prob = clf.predict_proba(X_test)

            # add to list
            prob_all_fold.append(pred_prob)
            label_all_fold.append(y_test)
        final_dict[k] = (prob_all_fold, label_all_fold, data_images[k])

    return final_dict

def linear_regression(train_loader, val_loader, feature_extractor, device):
    # data prep
    print("Preparing data...")
    data_features = []
    data_labels = []
    for img, label in tqdm(train_loader):
        features = feature_extractor(img, device, True)
        data_features.append(features.cpu().numpy())
        data_labels.append(label[0].cpu().numpy())
    data_features_train = np.concatenate(data_features, axis=0)
    data_labels_train = np.concatenate(data_labels, axis=0)

    data_features = []
    data_labels = []
    for img, label in tqdm(val_loader):
        features = feature_extractor(img, device, False)
        data_features.append(features.cpu().numpy())
        data_labels.append(label[0].cpu().numpy())
    data_features_val = np.concatenate(data_features, axis=0)
    data_labels_val = np.concatenate(data_labels, axis=0)

    # train linear regression
    print("Training starting ...")
    model = LinearRegression()
    model.fit(data_features_train, data_labels_train)
    print("Inference starting ...")
    # predict and return the labels
    pred_prob = model.predict(data_features_val)
    return pred_prob, data_labels_val

def linear_regression_with_cv(loader, feature_extractor, device):
    # data prep
    print("Preparing data...")
    data_features = []
    data_labels = []
    # with open("/h/richarddong/scratch/temp_results/gfp_visual/gfp_russian_retrain/alive_level.txt", "w") as f:
    #     for _, label in tqdm(loader):
    #         if label[3][0] < 0:
    #             print("Remove", label[1][0])
    #             try:
    #                 os.remove(label[1][0])
    #                 os.remove(label[1][0].replace("0.jpg", "1.jpg"))
    #                 os.remove(label[1][0].replace("0.jpg", "2.jpg"))
    #             except FileNotFoundError:
    #                 continue
    #         else:
    #             f.write(f"{label[1][0]},{label[3][0]}\n")
    for img, label in tqdm(loader):
        features = feature_extractor(img, device, True)
        data_features.append(features.cpu().numpy())
        data_labels.append(label[0].cpu().numpy())
    data_features = np.concatenate(data_features, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)

    # split to 10 fold
    print("Training starting ...")
    r_value = []
    total_pred = []
    total_label = []
    mae = []
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in tqdm(skf.split(data_features, data_labels)):
        X_train, X_test = data_features[train_idx], data_features[test_idx]
        y_train, y_test = data_labels[train_idx], data_labels[test_idx]

        # train the LR
        model = LinearRegression()
        model.fit(X_train, y_train)
    
        # compute R values
        pred_prob = model.predict(X_test)
        res = pearsonr(pred_prob.flatten(), y_test.flatten())

        # add to list
        r_value.append(res.statistic)
        total_pred.append(pred_prob)
        total_label.append(y_test)

    return r_value, total_pred, total_label

def mlp_with_cv(loader, feature_extractor, device, input_dim):
    # data prep
    print("Preparing data...")
    data_features = []
    data_labels = []
    for img, label in tqdm(loader):
        features = feature_extractor(img, device, True)
        data_features.append(features.cpu().numpy())
        data_labels.append(label[0].cpu().numpy())
    data_features = np.concatenate(data_features, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)

    # split to 10 fold
    print("Training starting ...")
    r_value = []
    total_pred = []
    total_label = []
    mae = []
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in tqdm(skf.split(data_features, data_labels)):
        X_train, X_test = data_features[train_idx], data_features[test_idx]
        y_train, y_test = data_labels[train_idx], data_labels[test_idx]

        # train the MLP
        model = Regression(input_dim=input_dim)
        model = build_model(model, task="regression")
        trainer = pl.Trainer(max_epochs=100, accelerator='auto', devices=1 if torch.cuda.is_available() else None, 
                            logger=False, enable_checkpointing=False)
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float().unsqueeze(1))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        trainer.fit(model, train_loader, val_loader)

        # compute R values
        model.eval()
        pred_prob = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                outputs = model(images.to(model.device))
                pred_prob.append(outputs.cpu().numpy())
        pred_prob = np.concatenate(pred_prob, axis=0)
        res = pearsonr(pred_prob.flatten(), y_test.flatten())

        # add to list
        r_value.append(res.statistic)
        total_pred.append(pred_prob)
        total_label.append(y_test)

    return r_value, total_pred, total_label

def visualizing_features(loader, feature_extractor, device, save_dir, name, class_to_idx=None):
    data_features = []
    data_labels = []
    if class_to_idx:
        rev_dic = {v:k for k,v in class_to_idx.items()}
    for img, label in tqdm(loader):
        features = feature_extractor(img, device, True)
        data_features.append(features.cpu().numpy())
        if class_to_idx:
            data_labels.append(np.array([rev_dic[label[0].cpu().numpy()[0]]]))
        else:
            data_labels.append(label[0].cpu().numpy())

    data_features = np.concatenate(data_features, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)

    np.savez_compressed(os.path.join(save_dir, f"{name}_features.npz"), features=data_features, labels=data_labels)

    # visualize with UMAP and plot with Altair
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(data_features)

    plt.figure(figsize=(10,10))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=data_labels, palette="tab20", s=5)
    plt.title(f"UMAP of features extracted by {name}")
    plt.savefig(os.path.join(save_dir, f"{name}_umap.png"))
    plt.close()

    # plot with Altair
    df = alt.Data(values=[{'UMAP1': float(embedding[i,0]), 'UMAP2': float(embedding[i,1]), 'label': str(data_labels[i])} for i in range(embedding.shape[0])])
    chart = alt.Chart(df).mark_circle(size=10).encode(
        x='UMAP1:Q',
        y='UMAP2:Q',
        color='label:N',
        tooltip=['label:N']
    ).interactive()
    chart.save(os.path.join(save_dir, f"{name}_umap.html"))

    # visualize pca and plot with Altair
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_features)
    df_pca = alt.Data(values=[{'PCA1': float(pca_result[i,0]), 'PCA2': float(pca_result[i,1]), 'label': str(data_labels[i])} for i in range(pca_result.shape[0])])
    chart_pca = alt.Chart(df_pca).mark_circle(size=10).encode(
        x=alt.X('PCA1:Q').title(f'PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})'),
        y=alt.Y('PCA2:Q').title(f'PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2%})'),
        color='label:N',
        tooltip=['label:N']
    ).interactive()
    chart_pca.save(os.path.join(save_dir, f"{name}_pca.html"))

    # compute pairwise distances in PCA between each label
    from sklearn.metrics import pairwise_distances
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['label'] = data_labels
    mean_pca = pca_df.groupby('label').mean().reset_index()
    distance_matrix = pairwise_distances(mean_pca[['PCA1', 'PCA2']])
    distance_df = pd.DataFrame(distance_matrix, index=mean_pca['label'], columns=mean_pca['label'])
    distance_df.to_csv(os.path.join(save_dir, f"{name}_pca_distance_matrix.csv"))

    # plot distance matrix as heatmap using altair
    # df_heatmap = alt.Data(values=distance_df.stack().reset_index().rename(columns={0: 'distance'}))
    df_heatmap = alt.Data(values=[{'label_x': str(mean_pca['label'][i]), 'label_y': str(mean_pca['label'][j]), 'distance': float(distance_matrix[i,j])} 
                                 for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1])])

    heatmap = alt.Chart(df_heatmap).mark_rect().encode(
        x=alt.X('label_x:N', title='Label'),
        y=alt.Y('label_y:N', title='Label'),
        color=alt.Color('distance:Q', title='Distance'),
        tooltip=['label_x:N', 'label_y:N', 'distance:Q']
    ).interactive()
    heatmap.save(os.path.join(save_dir, f"{name}_pca_distance_heatmap.html"))

    # compute the pairwise distances between all features
    pairwise_dist = pairwise_distances(data_features, metric='cosine')
    # save as npz
    np.savez_compressed(os.path.join(save_dir, f"{name}_pairwise_distances.npz"), distances=pairwise_dist, labels=data_labels)

def translation_stats(loader, conversion, device, save_dir, name, converted_path=None):
    '''
    Compute the SSIM, PSNR, and LPIPS between the original image and the image after translation
    '''
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity
    reshape = v2.Resize((256, 256))
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device) # other options: 'squeeze', 'vgg'
    ssim_list = []
    psnr_list = []
    lpips_list = []

    for img, labels in tqdm(loader):
        img = img.to(device)
        img = reshape(img)
        if converted_path:
            converted_img_path = labels[2][0]
            # replace the image path up to the last subdirectory with converted_path
            translated_path = os.path.join(converted_path, os.path.basename(os.path.dirname(converted_img_path)), os.path.basename(converted_img_path).replace(".tif", ".png"))
            translated_img = Image.open(translated_path).convert("RGB")
            translated_img = np.array(translated_img)
            if len(translated_img.shape) != 3:
                translated_img = np.stack([translated_img, translated_img, translated_img], axis=-1)
            to_tensor = v2.Compose([v2.ToImage(), 
                                v2.ToDtype(torch.float32),])
            translated_img = to_tensor(translated_img)
            translated_img = (translated_img - translated_img.min()) / max(translated_img.max()-translated_img.min(), 1e-5)
            translated_img = torch.clamp(translated_img, max=1, min=0) #ensure no float overflow
            translated_img = translated_img.unsqueeze(0).to(device)            
            translated_img = reshape(translated_img)
        else:
            with torch.no_grad():
                translated_img = conversion(img, device)
        # convert to gray scale for ssim
        gray_transform = v2.Grayscale(num_output_channels=3)
        img_grey = gray_transform(img)
        translated_img_grey = gray_transform(translated_img)
        ssim = ssim_metric(img_grey, translated_img_grey).item()
        psnr = psnr_metric(img_grey, translated_img_grey).item()
        lpips = lpips_metric(img_grey, translated_img_grey).item()
        # print(f"SSIM: {ssim}, PSNR: {psnr}, LPIPS: {lpips}")
        ssim_list.append(ssim)
        psnr_list.append(psnr)
        lpips_list.append(1 - lpips) # convert to similarity score by 1 - distance
    # save each as a txt file
    with open(os.path.join(save_dir, f"{name}_ssim.txt"), "w") as f:
        for s in ssim_list:
            f.write(f"{s}\n")
    with open(os.path.join(save_dir, f"{name}_psnr.txt"), "w") as f:
        for p in psnr_list:
            f.write(f"{p}\n")
    with open(os.path.join(save_dir, f"{name}_lpips.txt"), "w") as f:
        for l in lpips_list:
            f.write(f"{l}\n")
    print(f"Average SSIM: {np.mean(ssim_list)}, Average PSNR: {np.mean(psnr_list)}, Average 1 - LPIPS: {np.mean(lpips_list)}")

