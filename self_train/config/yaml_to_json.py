import yaml
import json


# yaml文件内容转换成json格式
def yaml_to_json(yamlPath):
    with open(yamlPath, encoding="utf-8") as f:
        datas = yaml.load(f, Loader=yaml.FullLoader)  # 将文件的内容转换为字典形式
    jsonDatas = json.dumps(datas, indent=5)  # 将字典的内容转换为json格式的字符串
    return jsonDatas


if __name__ == "__main__":
    yamlPath = 'deepspeed_zero3.yaml'
    with open(yamlPath.replace('.yaml', '.json'), 'w', encoding='utf-8') as f:
        datas = yaml_to_json(yamlPath)
        f.write(datas)
