from datasets import load_dataset

dataset = load_dataset(
    "eltorio/ROCOv2-radiology",
    split="train",
    streaming=True
)

print(type(dataset))
print(dataset.features)

sample = next(iter(dataset))

print("Fields:", sample.keys())
print("Caption:", sample["caption"])

import matplotlib.pyplot as plt

image = sample["image"]

print("Image type:", type(image))
print("Image size:", image.size)
print("Image mode:", image.mode)

plt.imshow(image)
plt.axis("off")
plt.title(sample["caption"])
plt.show()