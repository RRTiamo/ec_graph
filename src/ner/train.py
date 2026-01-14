# 使用Trainer定义训练脚手架
import time
from datasets import load_from_disk
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, EvalPrediction, EarlyStoppingCallback
from configuration.config import *

id2label = {id: label for id, label in enumerate(LABELS)}
label2id = {label: id for id, label in enumerate(LABELS)}

# 模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,
                                                        num_labels=len(LABELS),
                                                        id2label=id2label,
                                                        label2id=label2id
                                                        )

# 数据
train_dataset = load_from_disk(NER_PROCESSED_DATA_PATH / 'train')
valid_dataset = load_from_disk(NER_PROCESSED_DATA_PATH / 'valid')

# 超参数
args = TrainingArguments(
    # 训练轮次
    num_train_epochs=5,
    # 输出保存路径
    output_dir=str(CHECKPOINTS_DIR / NER),
    # 日志保存路径
    logging_dir=str(LOGS_DIR / NER / time.strftime("%Y-%m-%d-%H-%M-%S")),
    # 训练批次大小
    per_device_train_batch_size=BATCH_SIZE,
    # 验证批次大小
    per_device_eval_batch_size=BATCH_SIZE,
    # 评估策略
    eval_strategy='steps',
    # 评估步数
    eval_steps=SAVE_STEP,
    # 日志策略
    logging_strategy='steps',
    # 日志步数
    logging_steps=SAVE_STEP,
    # 保存策略
    save_strategy='steps',
    # 保存步数(检查点)
    save_steps=SAVE_STEP,
    # 保存个数限制
    save_total_limit=3,
    # 在最后加载最优模型
    load_best_model_at_end=True,
    # 使用fp16浮点数(混合精度计算)
    fp16=False,
    # 模型评价指标
    metric_for_best_model='eval_overall_f1',
    greater_is_better=True,
    # 加载检查点路径
    resume_from_checkpoint=str(CHECKPOINTS_DIR / NER)
)

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
        all_labels.append([id2label[id] for id in unpaid_labels])
        all_predictions.append([id2label[id] for id in unpaid_predictions])

    return seqeval.compute(predictions=all_predictions, references=all_labels)


# 早停
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3
)

# 训练器
trainer = Trainer(
    # 模型
    model=model,
    # 训练数据集
    train_dataset=train_dataset,
    # 评估数据集
    eval_dataset=valid_dataset,
    # 超参数
    args=args,
    # 数据收集器
    data_collator=data_collator,
    # 计算指标
    compute_metrics=compute_metrics,
    # 早停
    # callbacks=[early_stopping_callback]
)

# 训练
trainer.train()

# 保存模型
# 由于参数里传入了最后加载最优模型，所以这里可以直接保存
trainer.save_model(CHECKPOINTS_DIR / NER / 'best_model.pt')
