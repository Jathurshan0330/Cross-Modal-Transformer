# Towards Interpretable Sleep Stage Classification Using Cross-Modal Transformers

Accurate sleep stage classification is significant for sleep health assessment. In recent years, several deep learning and machine learning based sleep staging algorithms have been developed and they have achieved performance on par with human annotation. Despite improved performance, a limitation of most deep-learning based algorithms is their Black-box behavior, which which have limited their use in clinical settings. Here, we propose Cross-Modal Transformers, which is a transformer-based method for sleep stage classification. Our models achieve both competitive performance with the state-of-the-art approaches and eliminates the Black-box behavior of deep-learning models by  utilizing the interpretability aspect of the attention modules. The proposed cross-modal transformers consist of a novel cross-modal transformer encoder architecture along with a multi-scale 1-dimensional convolutional neural network for automatic representation learning. Our sleep stage classifier based on this design was able to achieve sleep stage classification performance on par with or better than the state-of-the-art approaches, along with interpretability, a fourfold reduction in the number of parameters and a reduced training time compared to the current state-of-the-art. This repository contains the implementation of epoch and sequence cross-modal transformers and the interpretations. 

![combined both models](https://user-images.githubusercontent.com/52663918/164946073-30f00556-b073-4aa0-8d52-460766a20a23.png)

## Getting Started

### Installation Guide
Run our algorithm using Pytorch and CUDA https://pytorch.org/

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```
pip install -r requirements.txt
```

## Train Cross-Modal Transformers

Train Epoch Cross-Modal Transformer

```
python cmt_training.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --train_data_list <train dataset fold as a list==> ex:[0,1,2,3]> --val_data_list <validation fold as a list==> ex:[4]> --model_type "Epoch" 
```

Train Sequence Cross-Modal Transformer

```
python cmt_training.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --train_data_list <train dataset fold as a list==> ex:[0,1,2,3]> --val_data_list <validation fold as a list==> ex:[4]>  --model_type "Seq" 
```

## Evaluate Cross-Modal Transformers

### Get Sleep Staging Results

Evaluate Epoch Cross-Modal Transformer

```
python cmt_evaluate.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --val_data_list <validation fold as a list==> ex:[4]> --model_type "Epoch" --batch_size 1
```

Evaluate Sequence Cross-Modal Transformer

```
python cmt_evaluate.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --val_data_list <validation fold as a list==> ex:[4]> --model_type "Seq" --batch_size 1
```

### Get Interpretations

The interpretation plots will be save under "./results/<give project name>/interpretations/<Data no>"
```
python cmt_evaluate.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --val_data_list <validation fold as a list==> ex:[4]> --model_type "Seq" --batch_size 1 --is_interpret True
```
  
![image](https://user-images.githubusercontent.com/52663918/164946114-2f84af8d-a49d-45e7-8836-c492c79e10b1.png)
  
![image](https://user-images.githubusercontent.com/52663918/164946126-218a7e1a-6569-4efc-8c3e-b4403497a5f0.png)

