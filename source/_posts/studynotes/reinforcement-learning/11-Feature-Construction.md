---
title: 11. Feature Construction
toc: true
date: 2020-06-15 08:09:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Feature Construction



참고: Coursera Reinforcement Learning (Alberta Univ.)

Value function을 $\hat{v}(s, w)$로 function approximation 한다고 하자. 이때, $s$는 어떤 state, $w$는 이 value function의 weight parameter이며, $\hat{v}(s, w)$는 value function을 parameterize한 함수이다.

만약, $\hat{v}(s, w)$가 linear model이라면, $\hat{v}(s, w) = x(s) \cdot w$가 될 것이다. 여기서, tabular method는 $x(s)$가 state개수만큼의 길이를 가진 one-hot vector였다. 즉, $x(s_i)$는 $i$번째 원소가 1이고 나머지는 0인 벡터였다.

하지만, 이것은 feature의 매우 한정적인 예시일 뿐, feature를 꼭 one-hot vector로 할 필요는 없을 것이다.

Feature를 제대로 선택하는 것은 매우 중요하다. Feature를 제대로 선택한다면, 간단한 linear model에서도 강력한 성능을 발휘할 가능성이 높다. 때로는 feature를 선택할 때, 어떤 분야의 domain 지식이 들어가기도 한다.

지금, feature를 어떻게 만들지에 대해서 정리하고자 한다.



## Coarse Coding

어떤 2차원 공간이 있다고 해 본다. 2차원 공간 속에서 어느 위치에 있는지를 하나의 state라고 해 본다. 그럼 state의 개수는 무한대이고, 이것을 tabular method로 나타내기는 불가능하다. 하지만, state aggregation을 한다면 이야기가 달라진다. 공간을 몇 등분으로 나누어서 각 공간에 들어가는 것을 하나의 상태로 표현하는 것이다.

![image-20200303011108051](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303011108051.png)

그리고, state aggregation은 기본적으로, 각 state가 겹치지 않는다. 또한, feature vector도 one-hot vector의 형태를 띄게 된다.

Coarse coding이란, aggregation을 겹치게 하도록 허용하는 방식으로, state aggregation를 조금 개선한 것이다.

![image-20200303011256015](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303011256015.png)

이때, agent는 동시에 여러개의 aggregation에 들어갈 수 있고, feature vector는 하나만 1인 벡터 형태가 아닌, 여러 원소가 1을 가지는 벡터 형태가 된다.

Coarse coding은 state aggregation의 generalization 형태라고 볼 수 있겠다.



### Generalization & Discrimination of Coarse Coding

Coarse coding에서, aggregation의 모양, 사이즈, 개수, 커버리지 범위 모양 등에 따라 generalization, discrimination이 크게 달라질 수 있다. aggregation이 겹친 상태에서 전체 state가 몇 개의 조각으로 나뉘는지 보면 discrimination이 얼마나 되는지 확인할 수 있다. 많은 개수로 나뉘면 discrimination이 많이 이뤄진다. 하나의 커버리지 범위가 넓고 개수가 많다면 discrimination이 또 증가하고, generalization또한 증가하는 경향이 있으나, 상황에 따라 다르기 때문에 적절히 결정할 필요가 있다.



## Tile Coding

Coarse coding의 한 종류로써, aggregation 모양이 그리드이다.

![image-20200303014339315](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303014339315.png)

이 그리드를 약간의 offset으로 여러번 움직인다.

![image-20200303014409374](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303014409374.png)

만약, 타일 1개가 넓으면, generalization도 상당히 넓어지고, offset의 크기와 방향, 그리드가 몇개냐에 따라 discrimination 정도도 달라진다. tile coding은 그리드로 나누고, 각 타일이 동일한 크기의 정/직사각형이기 때문에 현재 state가 어느 타일에 들어가는지 **매우 효율적으로 계산이 가능하다.**

다만, dimension이 증가할수록 tile개수는 exponential하게 증가하게 된다.



## Neural Network for Feature Construction

Neural network는 feature를 입력으로 받아서, hidden layer로 feature를 representation한 뒤, 최종 output을 낼 수 있는 복잡하고 유연한 parameterized function이다.

Neural network를 function approximation에 사용할 때의 장점은 feature construction을 인간이 아닌, 컴퓨터에게 맞김과 동시에 매우 유연하게 feature를 뽑아낼 수 있다는 것이다.

Neural network는 policy, value function, model 등, parameterized function이 필요한 모든 곳에 응용될 수 있다.