
# import os
# from bing_image_downloader.bing_image_downloader import downloader
# from utils import dataset_split

# # !git clone https://github.com/ndb796/bing_image_downloader

# def data_crawler(args, query):    
#     directory_lst = [os.path.join(args.root_path, args.data_path)+"/train/",os.path.join(args.root_path, args.data_path)+"/test/"]
#     downloader.download(query, limit=40,  output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
    # dataset_split(query, 30, directory_lst)