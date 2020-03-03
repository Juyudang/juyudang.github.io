---
title: 12. Prior Sensitivity Analysis
toc: true
date: 2020-03-01 22:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Prior Sensitivity Analysis



Prior sensitivity analysis란, 하나의 파라미터에 대해 여러가지 prior 분포를 적용해보고 만든 여러 모델의 성능을 분석하는 과정을 말한다.

여러가지 prior를 적용해서 여러 모델을 만들었다고 하자. 그럼 여러 모델들이 내놓는 결과를 바탕으로 다음과 같이 해석할 수 있다. (Prior sensitivity analysis의 결과로 다음 두 가지의 경우가 나온다)

- 결과가 prior-sensitive하다.

  어떤 prior를 선택하느냐에 따라 추정 성능 및 결과가 크게 달라지는 경우를 말한다. 즉, 데이터보다는 prior가 결과에 영향을 많이 미치는 경우이다. 이 경우, 내가 왜 이 prior를 선택해야 하고 이 모델을 선택해야 하는지 팀에게 설명할 필요가 있다.

- 결과가 prior-insensitive하다.

  이 경우, prior보다는 데이터가 결과에 지대한 영향을 미치는 경우로, prior의 선택에 큰 힘을 쏟을 필요가 없음을 보일 수 있다.



Prior sensitivity analysis는 내가 선택한 prior가 적절하다는 가정을 증명하기 위해서도 사용한다(즉, 가설검정에 사용할 수 있음). 내가 원하는 prior를 선택해서 모델을 구성하고, 내가 원하지 않는 prior를 선택해서 모델을 구성하게 된다. 이때, 내가 원하지 않는 prior를 **skeptical prior**라고 한다. 만약, skeptical prior로 시도해본 여러 모델들이 모두 내가 원한 prior 모델보다 일정 이상 성능이 좋지 않으면 나의 prior 선택을 증명 또는 설명할 수 있다.



Prior sensitivity analysis를 할때, 해당 prior로 posterior estimation을 통해 성능을 측정하게 된다.