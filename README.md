# ner_dataset_maker_by_gpt


한국어 ner 모델 파인튜닝을 위해 gpt-4o api를 통한 데이터셋 구축에 관한 저장소입니다.

소스코드 및 개념은 전적으로 [데이터 없이 NER 모델 학습하기](https://medium.com/@yongsun.yoon/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%97%86%EC%9D%B4-ner-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-90c4c24953a)의 내용을 바탕으로 하고 있으며 본 저장소에서는 api 호출파트 및 자잘한 소스코드의 수정만 진행하였습니다.

업로드 된 train.py에서는 gpt api를 통해 생성된 데이터셋을 바탕으로 [klue/roberta-small](https://huggingface.co/klue/roberta-small) 모델을 파인튜닝 하였습니다.

파인튜닝 결과는 다음 [허깅페이스 링크](https://huggingface.co/vitus9988/klue_roberta_small_ner_custom_domain)와 같습니다.
