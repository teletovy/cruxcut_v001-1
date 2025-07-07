#!/bin/bash

# gdown이 설치되어 있지 않다면 설치
pip install gdown --upgrade

# Google Drive 폴더 ID
FOLDER_ID="1IJysaxU1rpWRG8r1T-KCdr42gMLE4oAx"

# 전체 폴더 다운로드
gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID --remaining-ok
