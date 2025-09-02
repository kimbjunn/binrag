import json
import math
import copy
import re

def split_words(line):
    """
    snake_case와 camelCase를 모두 처리하여 단어를 분리하는 함수
    """
    if not isinstance(line, str):
        return []
    
    # 먼저 소문자로 변환하고 공백 제거
    line = line.strip()
    
    # camelCase 처리: 대문자 앞에 언더스코어 추가
    # 예: "camelCase" -> "camel_Case"
    line = re.sub('([a-z0-9])([A-Z])', r'\1_\2', line)
    
    # 연속된 대문자 처리: "HTTPRequest" -> "HTTP_Request"
    line = re.sub('([A-Z])([A-Z][a-z])', r'\1_\2', line)
    
    # 소문자로 변환
    line = line.lower()
    
    # 언더스코어로 분리
    words = line.split('_')
    
    # 빈 문자열 제거
    words = [word for word in words if word]
    
    return words

def get_correct_predictions_word_cluster(target, prediction, word_cluster):
    true_positive, false_positive, false_negative = 0, 0, 0
    replacement = dict()
    skip = set()
    for j, p in enumerate(prediction):
        if p in target:
            skip.add(j)
    for i, t in enumerate(target):
        for j, p in enumerate(prediction):
            if t != p and j not in replacement and j not in skip:
                if t in word_cluster and p in word_cluster:
                    t_cluster = word_cluster[t]
                    p_cluster = word_cluster[p]
                    t_cluster, p_cluster = set(t_cluster), set(p_cluster)
                    if len(t_cluster.intersection(p_cluster)) > 0:
                        replacement[j] = t
    for k, v in replacement.items():
        prediction[k] = v
    if target == prediction:
        true_positive = len(target)
    else:
        target = set(target)
        prediction = set(prediction)

        true_positive += len(target.intersection(prediction))
        false_negative += len(target.difference(prediction))
        false_positive += len(prediction.difference(target))
    return true_positive, false_positive, false_negative

def calculate_results(true_positive, false_positive, false_negative):
    if true_positive + false_positive == 0:
        return 0, 0, 0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1

def main():
    # 하드코딩된 JSON 파일 경로
    input_file = r"E:\symbol-infer\result\symgen_data\K\llama\10\output_20250710_183039_llama_k-10.json"
    word_cluster_path = r"E:\symbol-infer\embedding\kTrans-release-main\evaluation\word_cluster.json"

    # word_cluster 파일 로드
    with open(word_cluster_path, 'r', encoding='utf-8') as f:
        word_cluster = json.load(f)

    true_positive, false_positive, false_negative = 0, 0, 0
    total = 0
    targets = []
    predictions = []

    # JSON 데이터 읽기
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for idx, entry in enumerate(data):
            gt = entry['ground_truth']
            inf = entry['symbol']
            if not isinstance(gt, str) or not isinstance(inf, str):
                print(f"[디버그] {idx}번째 entry: {entry}")
                continue
                
            total += 1
            target = split_words(entry['ground_truth'])
            prediction = split_words(entry['symbol'])

            targets.append(entry['ground_truth'])
            predictions.append(entry['symbol'])
            
            # 디버깅을 위한 출력 (처음 5개만)
            if idx < 5:
                print(f"원본: '{entry['ground_truth']}' -> 분리: {target}")
                print(f"예측: '{entry['symbol']}' -> 분리: {prediction}")
                print("---")
            
            tp, fp, fn = get_correct_predictions_word_cluster(target, prediction, word_cluster)
            true_positive += tp
            false_positive += fp
            false_negative += fn

    precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
    
    # 파일에 결과 저장
    output_file = r"E:\symbol-infer\result\symgen_data\K\llama\10\output_20250710_183039_llama_k-10.txt"
    with open(output_file, 'w') as f_out:
        f_out.write("Total entries processed: {}\n".format(total))
        f_out.write("Precision: {:.4f}\n".format(precision))
        f_out.write("Recall: {:.4f}\n".format(recall))
        f_out.write("F1 Score: {:.4f}\n".format(f1))

    print(f"총 {total}개의 항목이 처리되었습니다.")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("평가 결과가 'output_20250705_062102_4.1-mini.txt'에 저장되었습니다.")

if __name__ == '__main__':
    main()