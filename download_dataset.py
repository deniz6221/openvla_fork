import gdown

url = 'https://drive.google.com/drive/folders/1VrSffmIQx6ccJglFoIh5CCNjJAIemi1w?usp=sharing'

gdown.download_folder(url, quiet=False, use_cookies=False)