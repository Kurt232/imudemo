# IMU-demo

## reference
1. \[2020\] LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications
2. \[2023\] IMU2CLIP: MULTIMODAL CONTRASTIVE LEARNING FOR IMU MOTION SENSORS FROM EGOCENTRIC VIDEOS AND TEXT NARRATIONS

## thoughts
1. traditional method for the task: unsupervised learning, like KNN is used to classify the unlabeled  data, and we use supervised learning to align the labels with the classes of data.  
2. With the rise of Transformer, like LIMU-BERT, we can use the unlabeled data to train a feature encoder to represent the data distribution by self-supervised learning(unsupervised learning), and we can fine-tune the model to align with the labels by supervised learning.  
![LIMU-BERT](assets/1714555504745.png)
3. With the development of GPT, it is reasonable to think a method like CLIP to represent two modals, IMU and text, using contrastive learning, and I find the IMU2CLIP. And I also find some researches using ChatGPT to generate textual description, like IMUGPT and HARGPT.
![IMU2CLIP](assets/20240501173214.png)
To some extent, the IMU2CLIP is also similar to the idea of Cosmos. I seem the dataset is bigger than Cosmos.
![Cosmos](assets/16851ab2179c6ea44bedd4b2b4c710a.png)

## dataset
hhar_20_120 from LIMU-BERT
```json
"hhar_20_120": {
        "sr": 20,
        "seq_len": 120,
        "dimension": 6,
        "activity_label_index": 2,
        "activity_label_size": 6,
        "activity_label": [ "bike", "sit", "downstairs", "upstairs", "stand", "walk"
        ],
        "user_label_index": 0,
        "user_label_size": 9,
        "model_label_index": 1,
        "model_label_size": 3,
        "size": 9166
    }
```

