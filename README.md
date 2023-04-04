# Towards Interpretable Sleep Stage Classification Using Cross-Modal Transformers

## Citation
If you find our work or this repository useful, please consider giving a star :star: and citation.
```bibtex
@article{pradeepkumar2022towards,
  title={Towards Interpretable Sleep Stage Classification Using Cross-Modal Transformers},
  author={Pradeepkumar, Jathurshan and Anandakumar, Mithunjha and Kugathasan, Vinith and Suntharalingham, Dhinesh and Kappel, Simon L and De Silva, Anjula C and Edussooriya, Chamira US},
  journal={arXiv preprint arXiv:2208.06991},
  year={2022}
}

```


## Abstract 

Accurate sleep stage classification is significant for sleep health assessment. In recent years, several deep learning and machine learning based sleep staging algorithms have been developed and they have achieved performance on par with human annotation. Despite improved performance, a limitation of most deep-learning based algorithms is their Black-box behavior, which which have limited their use in clinical settings. Here, we propose Cross-Modal Transformers, which is a transformer-based method for sleep stage classification. Our models achieve both competitive performance with the state-of-the-art approaches and eliminates the Black-box behavior of deep-learning models by  utilizing the interpretability aspect of the attention modules. The proposed cross-modal transformers consist of a novel cross-modal transformer encoder architecture along with a multi-scale 1-dimensional convolutional neural network for automatic representation learning. Our sleep stage classifier based on this design was able to achieve sleep stage classification performance on par with or better than the state-of-the-art approaches, along with interpretability, a fourfold reduction in the number of parameters and a reduced training time compared to the current state-of-the-art. This repository contains the implementation of epoch and sequence cross-modal transformers and the interpretations. 

![Epoch_CMT-1](https://user-images.githubusercontent.com/67052077/184390866-261038c3-4624-4857-872f-6d46c9c5363c.png)
![Seq_CMT-1](https://user-images.githubusercontent.com/67052077/184390916-1f5f811f-8416-4a62-8a52-a0e800144c7e.png)


## Getting Started

### Installation Guide
Run our algorithm using Pytorch and CUDA https://pytorch.org/

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```
pip install -r requirements.txt
```

## Dataset Generation

Kindly refer to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hs6V6doKpqlLlPOsnKllZp9ATuA9VqA7?usp=sharing) to extract dataset to train the models.

## Inference and Get Interpretations

Kindly refer to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cBDzzzwdyBp-bpxR_wvVzVlxrbSwrbiG?usp=sharing) to get prediction for any subject in sleepedf dataset and get interpretation results as given in the paper.


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

### Get Results

The interpretation plots will be save under "./results/<give project name>/interpretations/<Data no>"
```
python cmt_evaluate.py --project_path "./results/<give project name>" --data_path "path/to/dataset" --val_data_list <validation fold as a list==> ex:[4]> --model_type "Seq" --batch_size 1 --is_interpret True
```
 
  
 ## Sleep Stage Classification Results
  

![param_model_acc-1](https://user-images.githubusercontent.com/52663918/229873010-1d522a7b-fc5f-4a62-adf3-ac3e24d4cfe3.png)


## Interpretation Results
  
  ![33320_interpret-1](https://user-images.githubusercontent.com/67052077/184392262-1f85ea13-70a5-4d84-bb9e-491957e21929.png)

![44001_interpret-1](https://user-images.githubusercontent.com/67052077/184392276-c29553cb-9268-43d4-88ea-bfd8c1b20e0f.png)


