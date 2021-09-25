**참고사항

이 파일은 폴드에 있는 파일과 출력파일에 대한 설명이다.

* 폴드내 파일

- 학습데이터:
	- NIKL_CoLA_train.tsv
	- 15876 문장
- 개발데이터:
	- NIKL_CoLA_dev.tsv
	- 2032 문장 
- 평가데이터
	- NIKL_CoLA_test.tsv
	- 1060 문장 
- 평가 프로그램
	- eval.py


* 출력 파일의 형식
출력파일은 다음과 같은 형식을 가진다.

index<tab>acceptability_label<tab>sentence

acceptability_label은 0, 1 값을 가진다. 1은 문법에 올바른 문장을 나타내고 0은 그렇지 않은 문장을 나타낸다.
