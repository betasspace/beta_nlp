from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("cerebras/SlimPajama-627B", split="default", cache_dir="/Users/bytedance/workspace/test/multi_gpu/dataset/Slimpajama")
# bookcorpus.save_to_disk('/Users/bytedance/workspace/test/multi_gpu/dataset/Zhihu-KOL')

# bookcorpus = load_dataset("xuqinyang/BaiduBaike-5.63M", split="train")
# bookcorpus.save_to_disk('/Users/bytedance/workspace/test/multi_gpu/dataset/BaiduBaike')

# bookcorpus = load_dataset("zhengr/zhihu", split="train")
# bookcorpus.save_to_disk('/Users/bytedance/workspace/test/multi_gpu/dataset/zhihu_qa')

# ds = load_dataset("parquet", data_files="/Users/bytedance/workspace/test/multi_gpu/dataset/20220301.simple/train-00000-of-00001.parquet")
# ds = load_dataset("parquet", data_dir="/Users/bytedance/workspace/test/multi_gpu/dataset/20220301.simple/")