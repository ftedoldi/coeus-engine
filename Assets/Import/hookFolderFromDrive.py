import gdown

url = 'https://drive.google.com/drive/u/1/folders/12JN1tYfdxabtCbQ0thMZrkp1TML8EXkX'
gdown.download_folder(url, quiet=True, use_cookies=False)