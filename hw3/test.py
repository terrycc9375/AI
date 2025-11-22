# import torch

# print("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(0))
# print(torch.version.cuda)

# import transformers
# print(transformers.__version__)

# import random
# with open("./logs/test.txt", "a+") as f:
#     list_a = [random.random() for _ in range(10)]
#     list_b = [random.random() for _ in range(10)]
#     f.write(f"\na = {list_a}\nb = {str(list_b)}")

# import json
# history = list()
# with open("saved_models/summary.json", "r") as f:
#     history.extend(json.load(f))

# def flatten_to_list(data):
#     result = []
#     if isinstance(data, dict):
#         result.append(data)
#     elif isinstance(data, list):
#         for item in data:
#             result.extend(flatten_to_list(item))
#     return result

# history = flatten_to_list(history)

# with open("saved_models/summary.json", "w") as outf:
#     json.dump(history, outf, indent=4)
