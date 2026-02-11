from datasets import load_dataset

ds = load_dataset("camel-ai/chemistry", split="train", streaming=True)
ds = ds.take(50)
print(ds)
for sample in ds:
    print(sample)
