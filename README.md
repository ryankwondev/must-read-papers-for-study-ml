# Must Read Papers for ML Newbie (Updated Sep'23)

> 💡 머신러닝-딥러닝 입문자가 꼭 읽어야 할 논문 리스트

## Principles 

- [Stochastic Gradient Descent](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-23/issue-3/Stochastic-Estimation-of-the-Maximum-of-a-Regression-Function/10.1214/aoms/1177729392.full)
- [Error Backpropagation](https://www.nature.com/articles/323533a0)
- [Error Backpropagation through time(BPTT): ? et al.]()
- [Truncated Error Backpropagation through time(TBTT): ? et al.]()

## Components

- [Convolutional Neural Networks: Lecun et al.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [Recurrent Neural Networks: Romelhart et al.](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)
- [Long Shot Term Memory: Shumidhuber et al.](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Gated Recurrent Unit: Cho et al.](https://aclanthology.org/D14-1179.pdf)
- [Bahdanau Attention: Bahdanau et al.](https://arxiv.org/pdf/1409.0473.pdf)
- [Luong Attention: Luong et al.](https://aclanthology.org/D15-1166.pdf)
- [Self-Attention: Vaswani et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

## 컴퓨터 비전 (Computer Vision):

- [ImageNet Classification with Deep Convolutional Neural Networks(AlexNet): Krizhevsky et al.](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition(VGGNet): Simonyan et al.](https://arxiv.org/abs/1409.1556)
- [Going Deeper with Convolutions(GoogLeNet): Szegedy et al.](https://arxiv.org/abs/1409.4842)
- [Deep Residual Learning for Image Recognition(ResNet): He et al.](https://arxiv.org/abs/1512.03385)
- [You Only Look Once: Unified, Real-Time Object Detection(YOLO): Redmon et al.](https://arxiv.org/abs/1506.02640)
- [Mask R-CNN: He et al.](https://arxiv.org/abs/1703.06870)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale(Vision Transformer; ViT)](https://arxiv.org/pdf/2010.11929.pdf)
- [Masked Autoencoders are scalable Vision Learners: He et al.](https://arxiv.org/pdf/2111.06377.pdf)

## 자연어 처리 (Natural Language Processing, NLP):

- [Word2Vec: Mikolov et al.](https://arxiv.org/abs/1310.4546)
- [Bidirectional Encoder Representations from Transformers(BERT): Devlin et al.](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners(GPT-3): Brown et al.](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback(InstructGPT): Ouyang et al.](https://arxiv.org/abs/2203.02155)

## 음성 처리 (Speech Processing):

- [Deep Neural Networks for Acoustic Modeling in Speech Recognition(Deep Speech): Hinton et al.](https://ieeexplore.ieee.org/document/6296526)
- [A Generative Model for Raw Audio(Wavenet): van den Oord et al.](https://arxiv.org/abs/1609.03499)
- [Listen, Attend and Spell (LAS): Chan et al.](https://arxiv.org/abs/1508.01211)

## 강화 학습 (Reinforcement Learning):

- [Human-level control through deep reinforcement learning(Deep Q-Network; DQN): Mnih et al.](https://www.nature.com/articles/nature14236)
- [Combining Improvements in Deep Reinforcement Learning(Rainbow DQN): Hessel et al.](https://arxiv.org/abs/1710.02298)
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation(REINFORCE): Sutton et al.](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- [Continuous control with deep reinforcement learning(Deep Deterministic Policy Gradient; DDPG): Lilicrap et al.](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization Algorithms(Proximal Policy Optimization; PPO): Schulman et al.](https://arxiv.org/abs/1707.06347)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm: Silver et al.](https://arxiv.org/abs/1712.01815)
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model(MuZero): Schrittwieser et al.](https://arxiv.org/abs/1911.08265)


## Computational Optimisations

- [Asynchronous methods for deep reinforcement learning(Asynchronous Advantage Actor Critic; A3C): Mnih et al.](https://arxiv.org/abs/1602.01783)
- [Distributed Distributional Deterministic Policy Gradients(D4PG): Barth-Maron et al.](https://arxiv.org/abs/1804.08617)
- [Mastering Atari Games with Limited Data: Ye et al.](https://arxiv.org/abs/2111.00210)

<!--
## 컴퓨터 비전 (Computer Vision):
### 이미지 분류 (Image Classification): 이미지가 어떤 객체나 카테고리에 속하는지 분류합니다.
### 객체 검출 (Object Detection): 이미지 내에서 객체의 위치를 찾고 경계 상자를 그립니다.
### 얼굴 인식 (Face Recognition): 얼굴을 인식하고 개별 얼굴을 식별합니다.
### 이미지 분할 (Image Segmentation): 이미지를 픽셀 수준에서 객체로 분할합니다.
### 스타일 변환 (Style Transfer): 한 이미지의 스타일을 다른 이미지에 적용합니다.
### 자율 주행 자동차 (Autonomous Vehicles): 자율 주행 자동차에서 센서 데이터를 처리하고 환경을 이해하는 데 활용됩니다.

## 자연어 처리 (Natural Language Processing, NLP):
### 텍스트 분류 (Text Classification): 텍스트를 카테고리로 분류하거나 감정을 분석합니다.
### 기계 번역 (Machine Translation): 언어 간 번역을 수행합니다.
### 개체명 인식 (Named Entity Recognition, NER): 텍스트에서 명사나 개체를 식별합니다.
### 문서 요약 (Text Summarization): 긴 텍스트를 요약하여 핵심 내용을 추출합니다.
### 감정 분석 (Sentiment Analysis): 텍스트의 감정 톤을 분석합니다.
### 질의 응답 시스템 (Question Answering Systems): 질문에 대한 답변을 생성합니다.

## 음성 처리 (Speech Processing):
### 음성 인식 (Speech Recognition): 음성을 텍스트로 변환합니다.
### 음성 합성 (Speech Synthesis): 텍스트를 음성으로 변환합니다.
### 화자 인식 (Speaker Recognition): 특정 화자를 인식합니다.
### 음성 감정 분석 (Speech Emotion Analysis): 음성에서 감정을 분석합니다.

## 강화 학습 (Reinforcement Learning):
### 에이전트가 환경과 상호 작용하며 보상을 최대화하는 방법을 학습합니다.
### 게임, 로봇 제어, 금융 거래 등 다양한 응용 분야에서 사용됩니다.

## 생성적 모델 (Generative Models):
### 생성적 적대 신경망 (Generative Adversarial Networks, GANs): 이미지, 음성, 텍스트 등의 데이터를 생성합니다.
### 변이형 오토인코더 (Variational Autoencoders, VAEs): 데이터를 생성하고 분석하는 데 사용됩니다.

## 각종 응용 분야:
### 의료 이미지 분석 (Medical Image Analysis)
### 금융 예측 (Financial Forecasting)
### 화학 및 분자 모델링 (Chemistry and Molecular Modeling)
### 게임 개발 (Game Development)
### 로봇 공학 (Robotics)
### 환경 모니터링 (Environmental Monitoring)
-->


<!--
## CNN (Convolutional Neural Network) 

### 

### Computer Vision

- [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
-->

<!--
CNN (Convolutional Neural Network) 
LSTM (Long Short-Term Memory)
RNN (Recurrent Neural Network)
GAN (Generation Attemarical Network)
RBFN (Radial Basis Function Network)
MLP (Multi-Layer Perceptron)
SOM (Self Organization Map)
DBN (Deep Belief Networks)
RBM (Restricted Boltzmann Machine)
Autoencoder

ref: https://t.ly/othN4
-->
