# Evaluating-Student-Writing
![image](https://user-images.githubusercontent.com/69694512/204577662-6b2322a0-9504-49e8-9d55-74e36a10d307.png)

In this project, we’ll identify elements in student writing. More specifically, we will automatically segment texts and classify argumentative and rhetorical elements in essays written by 6th-12th grade students. 

Recommended models or libraries:**Roberta/Deberta/Longformer**

Data: The official training set has about **15,000 articles**, and the test set has about **10,000 articles**. Each element of the split is then categorized as one of the following: **Introduction / Position / Claim / Counterclaim / Rebuttal / Evidence / Concluding Statement** Note that some parts of the essay will be unannotated (i.e. they do not fit into the categories above).

Evaluation Criteria: **Coincidence between labels and predicted words** Calculate TP/FP/FN for each category and then take **macro F1 score** for all categories

# Data Explanation

* Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
* Position - an opinion or conclusion on the main question
* Claim - a claim that supports the position
* Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
* Rebuttal - a claim that refutes a counterclaim
* Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
* Concluding Statement - a concluding statement that restates the claims

# Coding Parts

## Modeling
```python
class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        # 载入 backbone
        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels) # 分类头
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        # 五个不同的dropout结果
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5 # 五层取平均
        return logits
```


## Post Processing
```python
# 每种实体的的最小长度阈值，小于阈值不识别
MIN_THRESH = {
    "I-Lead": 11,
    "I-Position": 7,
    "I-Evidence": 12,
    "I-Claim": 1,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

# 每种实体的的最小置信度，小于阈值不识别
PROB_THRESH = {
    "I-Lead": 0.687,
    "I-Position": 0.537,
    "I-Evidence": 0.637,
    "I-Claim": 0.537,
    "I-Concluding Statement": 0.687,
    "I-Counterclaim": 0.37,
    "I-Rebuttal": 0.537,
}
```

# Summary
1. longformer Baseline 5Fold;
2. Set the max_len of the reasoning stage to 4096;
3. Add post-processing;
4. Tried deberta-base but the score was too low, we did not try to add it to the fusion;
5. Deberta-large 5Fold added post-processing;
6. Fusion of the two models;
7. Adjust the learning rate, epoch, etc.
8. Corrected_train.csv after using the repaired label;
9. Try changing 5fold to 10fold;
10. Adjust parameters for post-processing;

