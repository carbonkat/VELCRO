from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed
import os 
import glob 

def process_image(image_path, size=(224, 224)):
    path = image_path.split('/')
    file_name = path[-1]
    full_path = '/'.join(path[1:-1])
    new_path = 'images-scaled-224/' + full_path + '/' + file_name
    try:
        img = Image.open(image_path).convert('RGB').resize(size, resample=Image.LANCZOS)
    except OSError:
        print("Error on file: " + image_path)
        return
    os.makedirs('images-scaled-224/' + full_path, exist_ok=True)
    img.save(new_path)

def main():
    os.makedirs('images-scaled-224', exist_ok=True)
    Parallel(n_jobs=-1)(delayed(process_image)(file) for file in tqdm(list(glob.glob('images/*/*/*.png')) + list(glob.glob('images/*/*.png')))) 
    

if __name__ == '__main__':
    main()
