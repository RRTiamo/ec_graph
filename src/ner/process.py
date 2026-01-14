from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from configuration.config import *

# 加载环境变量
load_dotenv(ROOT_PATH / '.env')


def process_data():
    # 加载数据
    dataset = load_dataset('json', data_files=NER_RAW_DATA_PATH)['train']

    # print(dataset)
    # features: ['text', 'id', 'label', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'],
    # 移除无关的列
    dataset = dataset.remove_columns(['id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'])

    # 划分数据集
    data_dict = dataset.train_test_split(0.2)
    data_dict['test'], data_dict['valid'] = data_dict['test'].train_test_split(0.5).values()

    # 定义分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 编码 is_split_into_words True已经分词，不会在分词
    def encode(example):
        tokens = list(example['text'])
        inputs = tokenizer(tokens, is_split_into_words=True, truncation=True)
        labels = [LABELS.index('O')] * len(tokens)
        for label in example['label']:
            start = label['start']
            end = label['end']
            # 标注数据
            labels[start:end] = [LABELS.index("B")] + [LABELS.index("I")] * (end - start - 1)
        # 将数据对齐，头尾加上cls和sep的token_id，好处是不参与计算也能让数据对齐
        labels = [-100] + labels + [-100]
        inputs['labels'] = labels
        return inputs

    # 由于无法对所有的text数据应用同一套label标签标注策略，不能使用批处理
    data_dict = data_dict.map(encode, remove_columns=['text', 'label'])

    # 保存数据集
    data_dict.save_to_disk(NER_PROCESSED_DATA_PATH)

    # print(data_dict['train'][0])


if __name__ == '__main__':
    process_data()
