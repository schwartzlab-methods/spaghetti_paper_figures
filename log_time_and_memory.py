import timeit
import tracemalloc
import argparse
import torch
import torchvision.transforms.v2 as v2
from utils.dataset import CustomImageDataset
from utils.utils import prep_datasets, loader_data
from utils.utils import image_transform as trans_cycleGAN
from feature_extractors import owkin_features, h_optimus_0_features, uni_features
from huggingface_hub import login
import timm
from downstream import init_cycleGAN, cycleGAN_transform
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import os
import torch

def main():
    print("cuda status: ", torch.cuda.is_available())
    transform_ops = [v2.RandomCrop(size=256//4), v2.Resize((256,256))]
    transform = v2.Compose(transform_ops)
    data = CustomImageDataset(args.image_path, transform=transform)
    class_to_idx = data.class_to_idx
    print("Data prep finished. The class to idx is: ", class_to_idx)
    if args.data_indices:
        indices = []
        for each in args.data_indices:
            with open(each, "r") as f:
                idx = f.readlines()
                idx = [int(x) for x in idx]
            indices.append(idx)
        _, data = prep_datasets(data, indices)
    total_loader = loader_data(data, batch_size=1)
    # prepare the feature extractor
    cycleGAN_gen = init_cycleGAN(args.convert)
    process_cycleGAN = trans_cycleGAN(do_augmentation=False, do_cropping=False)
    convert_to_HE = lambda x, device: cycleGAN_transform(cycleGAN_gen, device, process_cycleGAN, x)
    # the extractor list
    extractor_L = []
    # Phikon-v2
    feature_extractor = AutoModel.from_pretrained("owkin/phikon-v2")
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    feature_0 = lambda x, device, train: owkin_features(feature_extractor, device, image_processor, x, 
                                                    convert=convert_to_HE, invert=args.invert, do_filter=False, 
                                                    do_crop=False, train=train)
    extractor_L.append(feature_0)
    feature_1 = lambda x, device, train: owkin_features(feature_extractor, device, image_processor, x, 
                                                    convert=None, invert=False, do_filter=False, 
                                                    do_crop=False, train=train)
    extractor_L.append(feature_1)
    # h-optimus
    login(token=args.hugging_face_token)
    feature_extractor_2 = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    feature_2 = lambda x, device, train: h_optimus_0_features(feature_extractor_2, device, x, 
                                                            convert=convert_to_HE, invert=args.invert, do_filter=False,
                                                            do_crop=False, train=train)
    extractor_L.append(feature_2)
    feature_3 = lambda x, device, train: h_optimus_0_features(feature_extractor_2, device, x, 
                                                            convert=None, invert=False, do_filter=False,
                                                            do_crop=False, train=train)
    extractor_L.append(feature_3)
    # UNI
    feature_extractor_4 = timm.create_model(
                "hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    feature_4 = lambda x, device, train: uni_features(feature_extractor_4, device, x,
                                                    convert=convert_to_HE, invert=args.invert, do_filter=False,
                                                    do_crop=False, train=train)
    extractor_L.append(feature_4)
    feature_5 = lambda x, device, train: uni_features(feature_extractor_4, device, x,
                                                    convert=None, invert=False, do_filter=False,
                                                    do_crop=False, train=train)
    extractor_L.append(feature_5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # start the loop
    time_list = []
    peak_memory_list = []
    names = ["Phikon-v2", "Phikon-v2_no_convert", "H-optimus-0", "H-optimus-0_no_convert", "UNI", "UNI_no_convert"]
    for idx, (img, _) in enumerate(tqdm(total_loader)):
        # for large dataset, only do 1000 samples
        if idx >= 1000:
            break
        img = img.to(device)
        time_list_i = []
        peak_memory_list_i = []
        for i, feature in enumerate(extractor_L):
            start = timeit.default_timer()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                # mem_before = torch.cuda.memory_allocated(device)
            else:
                tracemalloc.start()
            _ = feature(img, device, train=False)
            if torch.cuda.is_available():
                # mem_after = torch.cuda.memory_allocated(device)
                # peak = mem_after - mem_before
                peak = torch.cuda.max_memory_allocated(device)
            else:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            print(f"Peak for {names[i]}: {peak / 10**6}")
            stop = timeit.default_timer()
            time_list_i.append(stop-start)
            peak_memory_list_i.append(peak / 10**6) # convert to MB
        time_list.append(time_list_i)
        peak_memory_list.append(peak_memory_list_i)
    # save the time and memory
    for i, name in enumerate(names):
        save_time = os.path.join(args.save_dir, f"{args.name}_{name}_time.txt")
        save_memory = os.path.join(args.save_dir, f"{args.name}_{name}_memory.txt")
        with open(save_time, "w") as f:
            for time in time_list:
                f.write(f"{time[i]}\n")
        with open(save_memory, "w") as f:
            for memory in peak_memory_list:
                f.write(f"{memory[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the exp")
    parser.add_argument("--image_path", type=str, nargs="+", help="Path to the image folder")
    parser.add_argument("--data_indices", type=str, nargs="+", help="Path to the data indices")
    parser.add_argument("--convert", type=str, help="Path to the cycleGAN model")
    parser.add_argument("--invert", action="store_true", help="Invert the image")
    parser.add_argument("--hugging_face_token", type=str, help="Hugging face token")
    parser.add_argument("--save_dir", type=str, help="Path to save the time and memory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    main()