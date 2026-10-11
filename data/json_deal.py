import json

#CD-ViTO对于ArTaxOr数据集的json命名在本项目中无法解析，需要重新进行id映射

# 加载JSON文件
with open('data/ArTaxOr/annotations/test.json', 'r') as f:
    data = json.load(f)

# 初始化计数器
image_id_counter = 1
annotation_id_counter = 1

# 创建一个映射，将旧的字符串id映射为新的整数id
image_id_map = {}

# 将images中的id转换为整数
for image in data['images']:
    image_id_map[image['id']] = image_id_counter  # 记录映射关系
    image['id'] = image_id_counter  # 更新为整数id
    image_id_counter += 1

# 将annotations中的image_id和id转换为整数
for annotation in data['annotations']:
    annotation['image_id'] = image_id_map[annotation['image_id']]  # 将image_id映射到新的整数id
    annotation['id'] = annotation_id_counter  # 给每个annotation一个新的整数id
    annotation_id_counter += 1

# 保存修改后的JSON文件
with open('data/ArTaxOr/annotations/fixed_test.json', 'w') as f:
    json.dump(data, f)

print("转换完成，保存为 'fixed_test.json'.")
