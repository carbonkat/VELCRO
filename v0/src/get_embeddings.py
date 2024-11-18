import pandas as pd
from glob import glob
import os
import torch
from model import mmContrastiveModel
from dataset import medicalDataModule
from tqdm import tqdm
import numpy as np

def main():
    print("loading model checkpoint")
    checkpoint_path = "/u1/l2hebert/src/src/l2hebert-logs/MedGeese/e3hpa1iv/checkpoints/epoch=10-step=17677.ckpt"
    model = mmContrastiveModel.load_from_checkpoint(checkpoint_path)
    hparams = model.hparams
    
    print("loading dataset checkpoint")
    print(model)
    print(hparams)
    clip_model = hparams.model_path
    clip_name = clip_model.split('/')[-1]
    dataset = medicalDataModule(model_path=clip_model, tensor_dir=clip_name + '-MedicalTensors-Final-organ/', batch_size=64, debug=False, force_remake=False, local_machine=False)
    print('DATASET PARAMS', dataset.hparams)
    dataset.prepare_data()
    dataset.setup("valid")
    print("lets dance!")
    dataloader = dataset.val_dataloader()
    print(dataloader)
    print(len(dataloader))
    model.eval()
    mask_embeds_list = []
    umls_embeds_list = []
    labels = []
    organs = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            model_batch = [x.to('cuda:0') for x in batch[:-1]]
            organ = batch[-1]
            masked_embeds, umls_embeds, loss, probs = model.step(model_batch)
 
            mask_embeds_list.append(masked_embeds.detach().cpu().numpy())
            umls_embeds_list.append(umls_embeds.detach().cpu().numpy())
            labels.append(batch[2].detach().cpu().numpy())
            organs.extend(organ)
    mask_embeds_list = np.concatenate(mask_embeds_list, axis=0)
    umls_embeds_list = umls_embeds_list[0]
    labels = np.concatenate(labels, axis=0)
    print(mask_embeds_list.shape)
    print(umls_embeds_list.shape)
    print(labels.shape)
    print(len(organs))
    with open('organ.txt', 'w') as f:
        for item in organs:
            f.write("%s\n" % item)
        
    np.save('mask_embeds.npy', mask_embeds_list)
    np.save('umls_embeds.npy', umls_embeds_list)
    np.save('labels.npy', labels)
    print("done!")


        
    ...

if __name__ == "__main__":
    main()