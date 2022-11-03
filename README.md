# Anomaly Detection using Unsupervised Learning


## Overview

interpolation 이나 padding으로 환자 마다 다른 수술 시간에 따른 데이터 크기 차이를 통일하고
전체를 딥러닝 모델의 input 으로 학습시켜 정상 환자와 급성 신손상 환자를 classification 해보았지만
정확도가 잘 나오지 않았다. 

그래서 정상 환자와 양성 환자의 전체 수술 중 반복되는 심전도 패턴 모양 자체가 아예 다른게
아니라 정상 환자와 양성 환자 모두 수술 중 심전도 데이터의 패턴 자체는 유사하지만 양성 환자에게 outlier 가 발생할 것이라고 가설을 세우게 되었다.

가설을 검증하고자 급성 신손상 환자의 심전도 데이터에 이상치가 존재하는 지 딥러닝을 통해 Anomaly Detection을 해보게 되었다.

## Anomaly Detection using Auto Encoder
![image](https://user-images.githubusercontent.com/79091824/199703294-d9016426-145e-4cd5-8ccb-25656d51b10b.png)

- Anomaly Detection에 내가 사용한 모델은  Auto Encoder 이다. Auto Encoder 로 정상 환자의 수술 중 심전도 데이터를 학습시킨 후에 정상환자의 수술 중 심전도 데이터와
양성 환자의 수술 중 심전도 데이터로 모델을 Test하여 Reconstruction Error 값을 계산하여 비교하였다. 양성 환자의 수술 중 심전도 데이터에 outlier가 있다면,
정상 환자의 수술 중 심전도 데이터학습한 모델의 input으로 들어갔을 때 Reconstruction Error가 크게 나오는 케이스이가있을 것으로 예상하였다.


## Raw Data
서울대 병원에서 제공하는 환자 데이터셋을 사용하였다.


## Data Preprocess

[논문](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb) 을 참고하여
심전도 데이터를 전처리하여 정상 환자 약 300명의 수술 중 심전도 데이터의 길이가 150인 샘플로 train set을 구축했고
정상환자 50명, 양성환자 50명의 수술 중 심전도 데이터로 test set을 구축했다.


## Models and Training Details

- Auto Encoder
- epoch은 50으로 하였고 optimizer는 Adam 으로 하였다

## Results

![image](https://user-images.githubusercontent.com/79091824/199701259-3ded573e-0bb7-4f33-99ef-0ee6d1e9361a.png)

- Reconstruction Error를 계산할 때, 환자마다 각각 수술 중 전체 심전도 데이터에 해당하는 샘플들을 input으로 하여 환자 마다 따로 Reconstruction Error를 계산한 후 
 최댓값을 구하여 정상 환자와 양성 환자의 Reconstruction Error 분포를 비교하였다.
- Reconstruction Error 값 자체르 보면 Auto Encoder 모델의 Train 은 잘 되어, input 시그널과 유사한 시그널이 Ouput으로 나온다고 생각하였다. 그러나
양성 환자와 정상환자의 Reconstruction Error 값 분포를 비교해본 결과, 두 분포 사이 큰 차이가 없어 내가 세웠던 가설이 틀렸다고 결론을 내리게 되었다.



## Reference 
* [Auto Encoder](https://arxiv.org/pdf/2003.05991.pdf)


