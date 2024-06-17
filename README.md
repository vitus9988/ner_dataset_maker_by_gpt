# NER Dataset Maker by GPT-4o

한국어 NER 모델 파인튜닝을 위한 데이터셋 구축 저장소입니다.

이 저장소는 [데이터 없이 NER 모델 학습하기](https://medium.com/@yongsun.yoon/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%97%86%EC%9D%B4-ner-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-90c4c24953a)의 개념을 기반으로 하고 있으며, API 호출 파트 및 일부 소스코드의 수정을 진행했습니다.

## 주요 구성 요소

### `create_dataset.py`

- `dataset_ko.json`에 명시된 엔티티 항목을 로드합니다.
- 이를 바탕으로 GPT API를 통해 짧은 한국어 문장을 생성합니다.
- 생성된 데이터를 CSV 파일로 저장합니다.

### `train.py`

- `create_dataset.py`의 실행 결과인 데이터셋 CSV 파일을 로드합니다.
- 본 예제 코드에서는 [klue/roberta-small](https://huggingface.co/klue/roberta-small) 모델을 파인튜닝합니다.
