## Single Image Debluring - Team DeFT

> 2023 Fall ML4ME

## References

- Image Deblur 관련 paper/repo 모음 [Link]('https://github.com/subeeshvasu/Awesome-Deblurring')

- High-quality Motion Deblurring from a Single Image, Qi Shan et al, 2008 [Matlab Code]('https://github.com/yangyangHu/deblur/tree/master')

## How to use

### Conda 환경 구축

- conda 설치는 [여기]('https://conda.io/projects/conda/en/latest/user-guide/install/index.html') 참고
- 가상환경 설정 처음이라면

```
conda env create -f ml4me.yml
conda activate ml4me
```

- 그 다음부터는

  - 가상환경 activate : `conda activate ml4me`
  - deactivate : `conda deactivate`

- 새로 라이브러리 설치했을 경우
  `conda list --export > ml4me.yaml`로 업데이트

### python 환경 구축

- 필요한 라이브러리 다운로드

```
pip install -r requirements.txt
```

### 그 외

- VSCode에 Git ID 등록

```
git config --global user.email "dkshin99@gmail.com"
git config --global user.name "DongkeyU99"
```
