from datasets import concatenate_datasets, load_dataset

SlimPajama = load_dataset("cerebras/SlimPajama-627B", cache_dir="/Volumes/data/dataset")

# wikipedia = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir="/Volumes/data/dataset")
# BaiduBaike = load_dataset("xuqinyang/BaiduBaike-5.63M", split="train", cache_dir="/Volumes/data/dataset")
# zhihu_qa = load_dataset("zhengr/zhihu", split="train", cache_dir="/Volumes/data/dataset")
