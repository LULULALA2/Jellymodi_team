<p align="center"><img src="https://user-images.githubusercontent.com/104331479/185329655-95f41df4-dec5-4e94-b6b8-471a0ef2deba.png" width="80%" height="80%"></p>

# <img src="https://user-images.githubusercontent.com/104331479/185330319-86af99b3-0eb2-4a75-a0c4-2b36808a3734.png" width="30" height="30"/> Project Jellymodi

**머신러닝 사물인식 프로젝트 - 젤리모디(Jellymodi: Jelly mood diary)**

Python OpenCV 를 이용하여 얼굴 표정을 인식하고, 분석한 표정에 해당하는 젤리 이모티콘을 넣어 일기를 쓰는 웹사이트 제작

<br>

# ⭐Intro
* mood tracker 기법을 사용하는 일기 작성 서비스
* Python OpenCV 을 이용한 얼굴인식
* 한국인 얼굴 표정 데이터 분석을 학습시킨 모델 제작 (데이터셋 : <a href="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82">AIHub - 한국인 감정인식을 위한 복합 영상</a>)
* S.A(Starting Assignment) : https://typingmylife.notion.site/Jellymodi-5e43c9f96bb04da7b4de26aac6eceeca

<div>
   <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
   <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=Flask&logoColor=white">
   <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=SQLite&logoColor=white">
</div>

<br>

# 📌Project
### 프로젝트 기획 및 목적
* 시연영상 : https://ddongkim.tistory.com/37
* 원본 팀 깃허브 : https://github.com/cmjcum/Jellymodi_team

* 목적 : backend 실력 향상 ( HTML, JavaScript, Ajax, Python, Flask, MongoDB )
* 1차 목표 : 인스타 페이지 기본기능 구현
  * 회원가입, 로그인, 로그아웃 기능 (JWT 사용)
  * 메인페이지 게시물(CRUD), 게시물 활동
  * 캠/사진을 통한 얼굴인식 → 머신러닝 사물인식 모델을 이용한 감정판별 → 해당 표정에 맞는 이모티콘 표시
* 2차 목표 : 부가기능 구현
  * 구글로그인 연동기능

<br>

### ✔ 담당 역할
<details>
<summary>한국인 감정인식 데이터셋 전처리 및 모델 전이학습</summary>      
<br>
기존 데이터셋에서는 라벨이 기쁨, 당황, 분노, 불안, 상처, 슬픔, 중립의 7개로 분류되어있었는데, 표정만 보고는 사람도 인식하기 힘든 표정이 많았기 때문에 인식률을 높이기 위하여 기쁨, 슬픔, 분노, 중립의 4개의 감정으로 카테고리를 줄였습니다. 또한 올바르게 분류되지 않은 사진이나 표정을 인식하기에 애매한 사진도 많았기 때문에 모델의 학습을 위해서 표정이 명확한 사진을 고르고 각 사진들에서 얼굴만 잘라내는 작업을 추가로 진행했습니다.
<br>
데이터 전처리 후에 만들어진 새로운 데이터셋으로 코랩에서 InceptionV3 모델을 전이학습하여 정확도 약 0.90의 감정분류 모델을 제작했습니다.
<br>(코랩 : https://colab.research.google.com/drive/1HKmdTvZSvowDDntpqg81ZUiUrp3hg4IY?usp=sharing)

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

input = Input(shape=(224, 224, 3))
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input, pooling='max')

x = base_model.output
x = Dropout(rate=0.25)(x)
x = Dense(64, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
```
</div>
</details>

- 회원가입/로그인 페이지, 글 작성 페이지 제작
- 회원가입/로그인 기능 구현

### ERD
![186601476-8fe8385d-8390-4747-9240-915795ca906c](https://user-images.githubusercontent.com/104331479/187063943-d493f725-7dc5-4240-8ee7-e348446cc377.png)


# 🧨TroubleShooting

### 1. 얼굴이 아닌 것을 얼굴로 인식하는 문제
책 표지나 포스터 등 얼굴이 아닌 것을 얼굴로 인식하는 문제가 발생했습니다. 사진 속의 얼굴 크기 평균치를 계산해서 얼굴 검출 사이즈를 평균크기 이상으로 한정시켜서 얼굴이 아닌 것을 얼굴로 인식하는 문제를 대부분 해결할 수 있었습니다.

<code> faces = face_cascade.detectMultiScale(gray, 1.2, minSize=(75,75)) </code>

### 2. 사진에서 얼굴만 잘라 저장하는 과정에서 계속 오류 발생
데이터 전처리과정에서 사진에서 얼굴만 잘라 저장해 모델을 학습시키려고 했습니다. 그런데 특정 사진은 얼굴인식도 제대로 되지 않고, 다음 사진으로도 넘어가지 않고 계속 오류로 멈추는 문제가 발생했습니다.

- 이미지 폴더 경로 수정 :  "\\"에서 "/"로 수정
- 특정 사진에서는 얼굴이 여러개 인식됨 : 얼굴 감지 크기에 최소 크기를 주고 여러개 인식되면 crop된 영역들을 번호를 붙여 모두 저장하게 함
- 특정 사진에서 얼굴이 인식되지 않음 : 인식이 안되서 멈추면 if문으로 crop된 사진이 1개 이상인 것만 넘어가게 함
- 코드가 돌아갔는데도 이미지가 저장이 안됨 : 저장 경로로 지정되있는 이미지 폴더를 미리 생성해줘야 함

<details>
<summary>코드 수정 전</summary> 

```python
import cv2
import glob

# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images = glob.glob('images\*')
for image in images:
    # 이미지 불러오기
    img = cv2.imread(image)
    img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 찾기
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img_x1 = x
        img_x2 = x + w
        img_y1 = y
        img_y2 = y + h

    img = img[img_y1:img_y2, img_x1:img_x2]

    try:
        print(f'face/{image}')
        cv2.imwrite(f'face/{image}', img)
    except:
        pass
```

</div>
</details>

<details>
<summary>코드 수정 후</summary> 

```python
import cv2
import glob

# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images = glob.glob('images/*')
print("images =", end=""), print(images)

result_list = []

for i in images:
    temp = i.replace("\\", "/")
    result_list.append(temp)

print("result_list = ", end=""), print(result_list)

# 이미지 불러오기
for image in result_list:

    if "jpg" in image:
        print("image = ", end=""), print(image)

        ori_img = cv2.imread(image)
        ori_img = cv2.resize(ori_img, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기
        faces = face_cascade.detectMultiScale(gray, 1.2, minSize=(75,75))
        print("faces =", end=""), print(faces)

        cnt = 0
        for (x, y, w, h) in faces:
            img_x1 = x
            img_x2 = x + w
            img_y1 = y
            img_y2 = y + h

            img = ori_img[img_y1:img_y2, img_x1:img_x2]

            image_file_name = image.split('.')
            print("image_file_name =", end=""), print(image_file_name)
            file_name = image_file_name[0]
            extension = image_file_name[1]

            file_name += str(cnt)

            filename = 'face/images/' + file_name + '.' + extension
            print("filename =", end=""), print(filename)
            cv2.imwrite(filename, img)

            cnt += 1
```

</div>
</details>

<br>

# 🖋회고
일기를 쓴다는 간단한 행위에 감정인식과 젤리이모티콘이라는 아이디어를 얹어서 재미있는 프로젝트로 만든 것 같아서 좋았다. 복잡한 기능을 넣어서 만든 프로젝트는 아니지만 처음으로 머신러닝을 프로젝트에 연결시켜 만들어본 프로젝트여서 뿌듯했다. 팀원들과 직접 데이터셋 정리부터 모델 제작까지 다 함께했던 것이 제일 재밌었던 일이 아닌가 싶다. 내가 쓴 코드 몇 줄에 모델이 학습을 해서 결과물을 보여준다는 것이 신기했다. 또한 매번 일주일이 안되는 짧은 시간동안 프로젝트를 진행하다보니 항상 시간에 쫓겨 프로젝트를 마감하고는 하는데, 노션으로 일정관리를 해보니 이번에는 좀 더 여유롭게 프로젝트를 마감할 수 있었던 것 같다. 짧은 시간이었지만 기획한대로 다 완성을 했고, 나중에 어플로 출시해보고 싶은 귀여운 모양새가 나와서 만족스러웠다.

<br>

# 👨‍👨‍👧‍👧TEAM Coumi
**팀 코우미 (코딩못하면 우는 미니언즈)**
| 팀원 | 이름 | 역할 | 깃허브 |
|:----------:|:----------:|:----------:|:----------:|
| 대장 미니언 (팀장) | **이정아** | 일기 조회 페이지 | <a href="https://github.com/zeonga1102"><img src="https://img.shields.io/badge/Github-000000?style=flat-square&logo=github&logoColor=white"/></a> |
| 디자이너 미니언 | **노을** | 메인 페이지 | <a href="https://github.com/minkkky"><img src="https://img.shields.io/badge/Github-000000?style=flat-square&logo=github&logoColor=white"/></a> |
| 기획자 미니언 | **이현경** | 로그인 및 회원가입 페이지 | <a href="https://github.com/LULULALA2"><img src="https://img.shields.io/badge/Github-000000?style=flat-square&logo=github&logoColor=white"/></a> |
| 막내 미니언 | **김동근** | 일기 작성 페이지 | <a href="https://github.com/cmjcum"><img src="https://img.shields.io/badge/Github-000000?style=flat-square&logo=github&logoColor=white"/></a> |
* 모델제작 - 팀 전체 참여

<br>

# ✏기획 및 일정
* 와이어프레임

![피그마 목업](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcoToUW%2FbtrCumtehoi%2FKCUpBnVBXOQhBOQnIMz4WK%2Fimg.png)

* 젤리티콘 제작

![젤리티콘제작](https://user-images.githubusercontent.com/104331479/185370310-7f77facf-19a5-43e2-8b67-f8bc4e62302d.PNG)

* 일정

![일정 관리](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdpoO9z%2FbtrCuVvoecp%2FV9ZiulP6KUSxxo8lWLfIG0%2Fimg.png)

<br>

# 실행화면
* 로그인 & 메인화면
![로그인-메인화면](https://user-images.githubusercontent.com/104331479/185368926-32a15726-98a1-47c1-a4a5-a5e025da8d36.png)

* 글작성
![글작성](https://user-images.githubusercontent.com/104331479/185368976-bee79add-1ff6-4aac-b344-394ab8a7ad08.png)

* 글 상세 & 메뉴
![글상세-모달](https://user-images.githubusercontent.com/104331479/185369005-ffd33b83-e248-4bb5-96ae-e0fc4d328218.png)
