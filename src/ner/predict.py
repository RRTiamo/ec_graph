import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from configuration.config import *


class Predict:
    def __init__(self, model, tokenizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, texts):
        # 判断是否是一条数据
        is_str = isinstance(texts, str)
        if is_str:
            texts = [texts]

        # 编码数据
        text_tokens = [list(text) for text in texts]
        input_tensors = self.tokenizer(text_tokens,
                                       is_split_into_words=True,
                                       padding=True,
                                       truncation=True,
                                       return_tensors='pt')
        with torch.no_grad():
            input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
            outputs = self.model(**input_tensors)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).tolist()
        final_predictions = []
        # 去除无关token
        for prediction, tokens in zip(predictions, text_tokens):
            remove_pad_predictions = prediction[1:len(tokens) + 1]
            # id2Token
            final_predictions.append(
                [self.model.config.id2label[id] for id in remove_pad_predictions])

        if is_str:
            return final_predictions[0]
        return final_predictions

    def extract(self, inputs):
        is_str = isinstance(inputs, str)
        all_entities = []
        if is_str:
            inputs = [inputs]
        predictions = self.predict(inputs)
        for input_text, predict in zip(inputs, predictions):
            entities = self.extract_entities(list(input_text), predict)
            all_entities.append(entities)
        if is_str:
            return all_entities[0]
        return all_entities

    @staticmethod
    def extract_entities(tokens, labels):
        current_entity = ""
        entity = []
        for token, label in zip(tokens, labels):
            if label == 'B':
                if current_entity:
                    entity.append(current_entity)
                current_entity = token
            elif label == "I":
                if current_entity:
                    current_entity += token
            else:
                if current_entity:
                    entity.append(current_entity)
                current_entity = ''
        # 最后一组bio
        if current_entity:
            entity.append(current_entity)
        return entity


def run_predict():
    model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')
    predictor = Predict(model, tokenizer)
    text = ["热风2018年秋季时尚女士运动风休闲鞋深口系带单鞋h11w8103",
            "麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩"]
    all_entities = predictor.extract(text)
    print(all_entities)


if __name__ == '__main__':
    run_predict()
