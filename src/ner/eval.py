import time
from datasets import load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, EvalPrediction, EarlyStoppingCallback
from configuration.config import *

# 加载最优模型
model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER / 'best_model.pt')

# 数据
test_dataset = load_from_disk(NER_PROCESSED_DATA_PATH / 'test')

# 数据收集器：做填充操作
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt'
)

# 评估函数
seqeval = evaluate.load('seqeval')


def compute_metrics(p: EvalPrediction):
    # batch_size,seq_len,num_class
    logits = p.predictions
    # id [ [], [], [] ]
    predictions = logits.argmax(axis=-1)
    labels = p.label_ids

    # label
    all_predictions = []
    all_labels = []
    for label, pred in zip(labels, predictions):
        # 去除无关Token
        unpaid_labels = label[label != -100]
        unpaid_predictions = pred[label != -100]
        # id2label
        all_labels.append([model.config.id2label[id] for id in unpaid_labels])
        all_predictions.append([model.config.id2label[id] for id in unpaid_predictions])

    return seqeval.compute(predictions=all_predictions, references=all_labels)


# 训练器
trainer = Trainer(
    # 模型
    model=model,
    # 评估数据集
    eval_dataset=test_dataset,
    # 数据收集器
    data_collator=data_collator,
    # 计算指标
    compute_metrics=compute_metrics,
)

# 训练
res = trainer.evaluate()
print(res)
