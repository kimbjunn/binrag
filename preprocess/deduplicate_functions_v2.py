import os
from typing import List, Tuple, Set
import re
import shutil

try:
    from tqdm import tqdm
except ImportError:
    print("[경고] tqdm 모듈이 설치되어 있지 않습니다. 진행률 표시를 원하면 'pip install tqdm'을 실행하세요.")
    def tqdm(x, *args, **kwargs):
        return x

def extract_functions_from_c_file(c_file_path: str) -> List[Tuple[str, str, str, str]]:
    """
    C 파일에서 함수별로 (함수명, 주소, 선언, 본문) 추출
    """
    with open(c_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(
        r'/\* Function: (.+?) at (0x[0-9A-Fa-f]+) \*/\s*(.*?)\n\{(.*?)\n\}',
        re.DOTALL
    )
    functions = []
    for match in pattern.finditer(content):
        func_name = match.group(1).strip()
        func_addr = match.group(2).strip()
        func_decl = match.group(3).strip()
        func_body = match.group(4).strip()
        functions.append((func_name, func_addr, func_decl, func_body))
    return functions

def normalize_func_signature_and_body(func_decl: str, func_body: str) -> tuple[str, str]:
    """
    함수 시그니처와 본문 내 sub_ 함수명을 FUN_x로 치환하고,
    dword_, qword_, word_, byte_ 패턴을 각각 개별적으로 정규화합니다.
    
    Args:
        func_decl: 함수 선언부
        func_body: 함수 본문
        
    Returns:
        정규화된 (함수 선언부, 함수 본문) 튜플
    """
    # 각 패턴별 매핑 딕셔너리와 카운터
    func_map: dict[str, str] = {}
    dword_map: dict[str, str] = {}
    qword_map: dict[str, str] = {}
    word_map: dict[str, str] = {}
    byte_map: dict[str, str] = {}
    
    func_count: int = 0
    dword_count: int = 0
    qword_count: int = 0
    word_count: int = 0
    byte_count: int = 0

    # 시그니처에서 sub_ 함수명 치환
    sig_match = re.search(r'\b(sub_[a-zA-Z0-9_]*)\s*\(', func_decl)
    if sig_match:
        func_name = sig_match.group(1)
        func_map[func_name] = f"FUN_{func_count}"
        norm_decl = re.sub(rf'\b{re.escape(func_name)}\b', func_map[func_name], func_decl, count=1)
        func_count += 1
    else:
        norm_decl = func_decl

    # 본문에서 함수 호출 치환 (괄호 포함)
    def func_replacer(match: re.Match) -> str:
        nonlocal func_count
        func_name = match.group(1)
        if func_name not in func_map:
            func_map[func_name] = f"FUN_{func_count}"
            func_count += 1
        return f"{func_map[func_name]}("
    
    norm_body = re.sub(r'\b(sub_[a-zA-Z0-9_]*)\s*\(', func_replacer, func_body)

    # 본문 내 함수 포인터/캐스팅/참조 등 괄호 없는 sub_도 치환
    for func_name in re.findall(r'\b(sub_[a-zA-Z0-9_]*)\b', norm_body):
        if func_name not in func_map:
            func_map[func_name] = f"FUN_{func_count}"
            func_count += 1
    norm_body = re.sub(
        r'\b(sub_[a-zA-Z0-9_]*)\b',
        lambda m: func_map.get(m.group(1), m.group(1)),
        norm_body
    )

    # dword_ 패턴 치환
    def dword_replacer(match: re.Match) -> str:
        nonlocal dword_count
        dword_name = match.group(1)
        if dword_name not in dword_map:
            dword_map[dword_name] = f"dword_{dword_count}"
            dword_count += 1
        return dword_map[dword_name]
    
    norm_decl = re.sub(r'\b(dword_[a-fA-F0-9]+)\b', dword_replacer, norm_decl)
    norm_body = re.sub(r'\b(dword_[a-fA-F0-9]+)\b', dword_replacer, norm_body)

    # qword_ 패턴 치환
    def qword_replacer(match: re.Match) -> str:
        nonlocal qword_count
        qword_name = match.group(1)
        if qword_name not in qword_map:
            qword_map[qword_name] = f"qword_{qword_count}"
            qword_count += 1
        return qword_map[qword_name]
    
    norm_decl = re.sub(r'\b(qword_[a-fA-F0-9]+)\b', qword_replacer, norm_decl)
    norm_body = re.sub(r'\b(qword_[a-fA-F0-9]+)\b', qword_replacer, norm_body)

    # word_ 패턴 치환
    def word_replacer(match: re.Match) -> str:
        nonlocal word_count
        word_name = match.group(1)
        if word_name not in word_map:
            word_map[word_name] = f"word_{word_count}"
            word_count += 1
        return word_map[word_name]
    
    norm_decl = re.sub(r'\b(word_[a-fA-F0-9]+)\b', word_replacer, norm_decl)
    norm_body = re.sub(r'\b(word_[a-fA-F0-9]+)\b', word_replacer, norm_body)

    # byte_ 패턴 치환
    def byte_replacer(match: re.Match) -> str:
        nonlocal byte_count
        byte_name = match.group(1)
        if byte_name not in byte_map:
            byte_map[byte_name] = f"byte_{byte_count}"
            byte_count += 1
        return byte_map[byte_name]
    
    norm_decl = re.sub(r'\b(byte_[a-fA-F0-9]+)\b', byte_replacer, norm_decl)
    norm_body = re.sub(r'\b(byte_[a-fA-F0-9]+)\b', byte_replacer, norm_body)

    return norm_decl, norm_body

def deduplicate_functions(functions: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
    """
    정규화된 함수 본문 + 함수명 기준 중복 제거
    (동일한 함수 본문을 가지더라도 함수명이 다르면 모두 남김)
    """
    existed_func_key: Set[Tuple[str, str]] = set()  # (정규화된 본문, 함수명)
    unique_funcs = []
    for func_name, func_addr, func_decl, func_body in functions:
        norm_decl, norm_body = normalize_func_signature_and_body(func_decl, func_body)
        func_key = (norm_body, func_name)
        if func_key in existed_func_key:
            continue
        existed_func_key.add(func_key)
        unique_funcs.append((func_name, func_addr, func_decl, func_body))
    return unique_funcs

def copy_non_c_files(input_dir: str, output_dir: str) -> None:
    """
    입력 디렉토리에서 .c 파일을 제외한 모든 파일을 출력 디렉토리로 동일한 구조로 복사합니다.
    """
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith('.c'):
                continue
            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(root, input_dir)
            dst_dir = os.path.join(output_dir, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, fname)
            shutil.copy2(src_path, dst_path)

def main():
    input_dir = r"E:\symbol-infer\dataset\malware\MalwareSourceCode\dataset\strip_noZN_deduplicated"
    output_dir = r"E:\symbol-infer\dataset\malware\MalwareSourceCode\dataset\strip_noZN_deduplicated_v2"

    all_functions = []
    file_func_map = {}

    # 1차: 파일별 함수 추출 진행률 표시
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith('.c'):
                continue
            c_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, input_dir)
            rel_path = os.path.join(rel_dir, fname)
            file_list.append((c_path, rel_path))

    for c_path, rel_path in tqdm(file_list, desc="[1/3] 함수 추출 중", unit="file"):
        functions = extract_functions_from_c_file(c_path)
        all_functions.extend(functions)
        file_func_map[rel_path] = functions

    # 2차: 중복 제거 진행률 표시
    print(f"[2/3] 함수 중복 제거 중... (총 {len(all_functions)}개 함수)")
    unique_funcs = deduplicate_functions(all_functions)
    unique_func_keys = set((name, addr) for name, addr, _, _ in unique_funcs)

    # 3차: 파일별 저장 진행률 표시
    for rel_path, functions in tqdm(file_func_map.items(), desc="[3/3] 파일 저장 중", unit="file"):
        filtered = [
            (name, addr, decl, body)
            for name, addr, decl, body in functions
            if (name, addr) in unique_func_keys
        ]
        if not filtered:
            continue
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for name, addr, decl, body in filtered:
                norm_decl, norm_body = normalize_func_signature_and_body(decl, body)
                f.write(f"/* Function: {name} at {addr} */\n")
                f.write(f"{norm_decl}\n")
                f.write("{\n")
                f.write(f"{norm_body}\n")
                f.write("}\n\n")
    print(f"중복 제거 및 파일 저장 완료: {len(unique_funcs)}개 함수")
    copy_non_c_files(input_dir, output_dir)

if __name__ == '__main__':
    main() 