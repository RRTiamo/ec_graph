from transformers import AutoModelForTokenClassification, AutoTokenizer

from datasync.utils import MysqlReader, Neo4jWriter
from ner.predict import Predict
from configuration.config import *


class TextSync:
    def __init__(self):
        self.reader = MysqlReader()
        self.writer = Neo4jWriter()
        self.extractor = self._init_extractor()

    @staticmethod
    def _init_extractor():
        model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')
        return Predict(model, tokenizer)

    def sync_tag(self):
        # 获取数据
        sql = """
              select id, description
              from gmall.spu_info
              """
        items = self.reader.read(sql)
        ids = [item['id'] for item in items]
        descriptions = [item['description'] for item in items]
        # 提取标签
        all_entities = self.extractor.extract(descriptions)

        properties = []
        relations = []
        for id, entities in zip(ids, all_entities):
            # print(id,entity)
            for index, entity in enumerate(entities):
                tag_id = "-".join([str(index), str(id)])
                tag_property = {"id": tag_id, "name": entity}
                properties.append(tag_property)
                relation = {"start_id": id, "end_id": tag_id}
                relations.append(relation)
        # 写入节点和关系
        self.writer.write_nodes("TAG", properties)
        self.writer.write_relations("Have", "SPU", "TAG", relations)


if __name__ == '__main__':
    text_sync = TextSync()
    text_sync.sync_tag()
