# 병렬 설치는 안 되지만, 큰 패키지들 먼저 설치
poetry add torch torchvision  # 가장 오래 걸리는 것들 먼저

# 그 다음 나머지
while IFS= read -r line; do
    package=$(echo "$line" | cut -d'=' -f1)
    if [[ "$package" != "torch" && "$package" != "torchvision" ]]; then
        poetry add "$line"
    fi
done < requirements.txt