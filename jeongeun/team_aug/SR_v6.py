# spacing
# pip install git+https://github.com/haven-jeon/PyKoSpacing.git

# 형태소 분석기 설치 필요
# pip install konlpy


import pickle
import pandas as pd
from konlpy.tag import Okt
from pykospacing import Spacing
from tqdm.auto import tqdm


# 동의어 교체 함수
def sr_augment(data_path, save_path, wordnet_path):

    # 교체 문장 생성 함수
    def s_maker(sentence, common, synonym):

        # 조사 교정 함수
        def change_josa(noun, josa):

            # 한글의 유니코드가 28로 나누어 떨어지면 받침이 없음
            if josa == "이" or josa == "가":
                return "이" if (ord(noun[-1]) - ord("가")) % 28 == 0 else "가" # 책이 / 사과가
            elif josa == "을" or josa == "를":
                return "을" if (ord(noun[-1]) - ord("가")) % 28 == 0 else "를" # 책을 / 사과를
            elif josa == "과" or josa == "와":
                return "과" if (ord(noun[-1]) - ord("가")) % 28 == 0 else "와" # 책과 / 사과와
            else:
                return josa

        spacing = Spacing()

        replace_sentence = []

        # 조사 체크
        check = set(['이', '가', '을', '를', '과', '와'])

        for i in range(len(sentence)):
            # 문장에서 동의어 추가
            if sentence[i][0] == common:
                replace_sentence.append(synonym)

                # 뒷말이 조사면 조사 교정
                if i + 1 < len(sentence) and sentence[i+1][1] == 'Josa' and sentence[i+1][0] in check:
                    sentence[i + 1] = (
                    change_josa(replace_sentence[-1][0], sentence[i + 1][0]),
                    'Josa',
                )
            else:
                replace_sentence.append(sentence[i][0])

        # 형태소 연결 및 문장 생성
        replace_sentence = "".join(replace_sentence)
        replace_sentence = spacing(replace_sentence)

        return replace_sentence

    # 동의어 사전 불러오기
    with open(wordnet_path, "rb") as f:
        wordnet = pickle.load(f)

    # 데이터 불러오기
    data = pd.read_csv(data_path)
    s1, s2 = data['sentence_1'], data['sentence_2']

    okt = Okt() # Okt -> 한국어 텍스트에서 명사, 동사, 형용사 등을 추출

    sr_sentence = []

    for i in tqdm(range(len(s1)), desc='sr_sentence'):

        # i번째 문장에서 명사만 추출
        n1 = okt.nouns(s1[i])
        n2 = okt.nouns(s2[i])

        # s1과 s2의 공통된 명사를 추출
        common = set(n1) & set(n2)

        for com in common:

            # 길이가 2 이상 (분석기 오류 예방), wordnet에 있는지 확인
            if len(com) >=2 and com in wordnet and len(wordnet[com]) >= 2:
                # 동의어 추출
                synonym = wordnet[com][1]

                # 형태소 단위로 분리 및 품사 태그와 함께 추출
                s1_tag = okt.pos(s1[i])
                s2_tag = okt.pos(s2[i])

                # 동의어 변환 및 추가
                sr_sentence.append([
                    data['id'][i],
                    data['source'][i],
                    s_maker(s1_tag, com, synonym),
                    s_maker(s2_tag, com, synonym),
                    data['label'][i],
                    data['binary-label'][i],
                ])

    # list -> df
    sr_sentence = pd.DataFrame(
        sr_sentence,
        columns=['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label']
    )

    # label 값이 1보다 크거나 같은 것만 추가해서 저장
    sr_sentence = sr_sentence[sr_sentence['label'] >= 1]

    result = pd.concat([data, sr_sentence])
    result.to_csv(save_path, index=False)


sr_augment("train_all.csv", "SR_train_6.csv", "wordnet.pickle") # wordnet.pickle -> KAIST에서 만든 Korean WordNet(KWN)
