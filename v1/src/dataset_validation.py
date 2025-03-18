import os
from glob import glob
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage
import numpy as np

tensor_dir = "./data/dataset_files//processed_tensors/"
examples = list(glob(tensor_dir + "/masks/*.pt"))
print("starting checking...")
count = 0
sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to("cuda:0")
sam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
for i, point in enumerate(examples):
    if i%10000 == 0:
        print(i)
    #count+=1
    #if '47783' not in point:
    #    continue
    (mask, label, candidate_text, bb, _) = torch.load(
        point, weights_only=False
    )
    check_mask = mask.detach().numpy()
    #print(len(list(np.unique(mask))))
    #break
    path = point
    split_basename = os.path.basename(path).split("-")
    stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
    tensor_dir = os.path.dirname(os.path.dirname(path))
    try:
        #img = torch.load(tensor_dir + f"/images/{stem}", weights_only=False)
        #print(torch.unique(mask).shape[0])
        valid_mask = (len(list(np.unique(check_mask))) >= 2)
        print(valid_mask)
        if not valid_mask:
            #print(point)
            count += 1
            #inputs = sam_processor(img, input_boxes=[[bboxes]], return_tensors="pt").to("cuda")
            #outputs = sam_model(**inputs, multimask_output=False)
            #segment_mask = sam_processor.image_processor.post_process_masks(
            #    outputs.pred_masks.cpu(),
            #    inputs["original_sizes"].cpu(),
            #    inputs["reshaped_input_sizes"].cpu(),
            #    binarize=True,
            #)[0]
            #segment_mask = segment_mask.squeeze().numpy()
            #segment_mask = Image.fromarray(segment_mask)
            #assert segment_mask.size == (224, 224), segment_mask.size
            #segment_mask = segment_mask.convert("RGB")
            #segment_mask = PILToTensor()(segment_mask)
            #if torch.unique(segment_mask).shape[0] < 2:
            #    count += 1
            #assert len(list(np.unique(segment_mask.detach().numpy()))) >= 2, f"expected >= 2 unique values but got < 2 instead"
        torch.save(
            (mask, label, candidate_text, bb, valid_mask),
            (path),
        )
    except Exception as e:
        print(e)
        print(point)
        #print(f"Error on file when loading: {stem}")
        #if os.path.exists(tensor_dir + f"/images/{stem}"):
        #    print("File exists")
        #else:
        #    print("File does not exist")
print("finished checking!")
print(f"{count} masks failed to be generated!")
print(f"{count/len(examples)} of masks out of all masks failed to be generated!")
