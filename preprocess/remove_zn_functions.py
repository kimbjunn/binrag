import os
import re
import shutil
from typing import Optional, List, Tuple

def edit_file(target_file: str, instructions: str, code_edit: str) -> None:
    """
    파일을 수정합니다.
    """
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(code_edit)

def is_zn_function(name: str) -> bool:
    """
    함수 이름이 ZN 함수인지 확인합니다.
    
    Args:
        name (str): 함수 이름
        
    Returns:
        bool: ZN 함수 여부
    """
    # 컴파일된 정규식 패턴 (성능 최적화)
    ZN_PATTERNS = [
        re.compile(r'_ZN'),   # _ZN 포함
        re.compile(r'\._ZN'),  # ._ZN 포함
        re.compile(r'_Z'),    # _Z 포함
        re.compile(r'\._Z'),  # ._Z 포함
    ]
    
    # 디버깅을 위한 출력 추가
    print(f"검사 중인 함수 이름: {name}")
    
    return any(pattern.search(name) for pattern in ZN_PATTERNS)  # match 대신 search 사용

def find_function_blocks(content: str) -> List[Tuple[int, int, str]]:
    """
    파일 내용에서 함수 블록을 찾아 반환합니다.
    
    Args:
        content (str): 파일 내용
        
    Returns:
        List[Tuple[int, int, str]]: [(시작 위치, 끝 위치, 함수 이름)]
    """
    blocks = []
    # 컴파일된 정규식 패턴 (성능 최적화)
    comment_pattern = re.compile(r'/\* Function: (.*?) at (0x[0-9A-Fa-f]+) \*/')
    
    # 모든 함수 주석 위치 찾기
    comments = list(comment_pattern.finditer(content))
    
    # 각 주석 사이의 내용을 함수 블록으로 처리
    for i, match in enumerate(comments):
        # 함수 이름 추출 (첫 번째 그룹)
        func_name = match.group(1).strip()  # strip()으로 앞뒤 공백 제거
        start = match.start()
        # 마지막 주석이면 파일 끝까지, 아니면 다음 주석 전까지
        end = comments[i + 1].start() if i + 1 < len(comments) else len(content)
        blocks.append((start, end, func_name))
        
    return blocks

def remove_zn_functions_from_c_file(c_file_path: str, output_path: str, backup: bool = True) -> tuple[bool, int]:
    """
    .c 파일에서 함수 이름에 'ZN'이 포함된 함수 전체(주석, 선언, 본문)를 제거한다.

    Args:
        c_file_path (str): 대상 C 파일 경로
        output_path (str): 결과를 저장할 파일 경로
        backup (bool): 원본 백업 여부

    Returns:
        tuple[bool, int]: (실제로 변경이 발생했으면 True, 아니면 False, 제거된 함수 개수)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
    """
    if not os.path.isfile(c_file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {c_file_path}")

    # 파일을 한 번에 읽기 (성능 최적화)
    with open(c_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 함수 블록 찾기
    blocks = find_function_blocks(content)
    
    # 디버깅을 위한 출력 추가
    print(f"\n발견된 함수 블록:")
    for start, end, name in blocks:
        print(f"- {name}")
    
    # 제거할 함수 블록 찾기 (ZN 함수 패턴 개선)
    blocks_to_remove = [(start, end, name) for start, end, name in blocks if is_zn_function(name)]
    
    # 디버깅을 위한 출력 추가
    print(f"\n제거할 함수 블록:")
    for start, end, name in blocks_to_remove:
        print(f"- {name}")
    
    if not blocks_to_remove:
        print("제거할 함수가 없습니다.")
        return False, 0
        
    # 새 내용 생성 (성능 최적화)
    new_content = []
    last_end = 0
    
    for start, end, name in blocks_to_remove:
        print(f"제거 중: {name} ({start}:{end})")
        new_content.append(content[last_end:start])
        last_end = end
        
    new_content.append(content[last_end:])
    
    # 파일 저장
    if backup:
        try:
            shutil.copy2(c_file_path, output_path + '.bak')
        except Exception as e:
            print(f"백업 파일 생성 실패: {e}")
            print("백업 없이 계속 진행합니다.")
    
    # 문자열 연결 최적화
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(new_content))
    except Exception as e:
        print(f"파일 저장 실패: {e}")
        return False, 0
        
    return True, len(blocks_to_remove)

def process_directory_remove_zn(root_dir: str, output_dir: str, backup: bool = True) -> None:
    """
    지정한 디렉토리(재귀) 내 모든 .c 파일에서 'ZN'이 포함된 함수 전체를 제거한다.

    Args:
        root_dir (str): 루트 디렉토리 경로
        output_dir (str): 결과를 저장할 디렉토리 경로
        backup (bool): 백업 여부
    """
    changed_files = 0
    total_files = 0
    total_functions_removed = 0
    total_functions = 0

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 파일 목록 미리 수집 (성능 최적화)
    c_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.c'):
                c_files.append(os.path.join(dirpath, filename))
    
    total_files = len(c_files)
    
    # 진행 상황 표시를 위한 tqdm 사용
    try:
        from tqdm import tqdm
        iterator = tqdm(c_files, desc="파일 처리 중")
    except ImportError:
        iterator = c_files

    for c_file_path in iterator:
        try:
            # 상대 경로 계산
            rel_path = os.path.relpath(c_file_path, root_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # 출력 파일의 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 파일의 총 함수 개수 계산 (성능 최적화)
            with open(c_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                total_funcs_in_file = len(re.findall(r'/\* Function: .*? at 0x[0-9A-Fa-f]+ \*/', content))
                total_functions += total_funcs_in_file

            # 함수 제거 실행
            changed, removed_count = remove_zn_functions_from_c_file(c_file_path, output_path, backup=backup)
            if changed:
                print(f"[제거됨] {c_file_path} -> {output_path}")
                print(f"  - 총 함수 개수: {total_funcs_in_file}")
                print(f"  - 제거된 함수: {removed_count}개")
                print(f"  - 남은 함수: {total_funcs_in_file - removed_count}개")
                changed_files += 1
                total_functions_removed += removed_count
            else:
                # ZN 함수가 없으면 원본 파일을 그대로 복사
                shutil.copy2(c_file_path, output_path)
                print(f"[복사됨] {c_file_path} -> {output_path} (ZN 함수 없음)")
        except Exception as e:
            print(f"[오류] {c_file_path}: {e}")

    print(f"\n처리 결과:")
    print(f"- 총 {total_files}개 .c 파일 처리")
    print(f"- {changed_files}개 파일에서 함수 제거 발생")
    print(f"- 총 {total_functions}개 함수 중 {total_functions_removed}개 함수 제거")
    print(f"- 제거율: {(total_functions_removed / total_functions * 100):.1f}%")

if __name__ == '__main__':
    import sys
    print("함수명에 'ZN'이 포함된 함수 전체를 제거할 디렉토리 경로를 입력하세요:")
    path = input("입력 디렉토리 경로: ").strip()
    if not os.path.isdir(path):
        print("디렉토리가 존재하지 않습니다.")
        sys.exit(1)
        
    print("결과를 저장할 디렉토리 경로를 입력하세요:")
    output_path = input("출력 디렉토리 경로: ").strip()
    
    try:
        process_directory_remove_zn(path, output_path)
        print("작업이 완료되었습니다. (수정된 파일은 .bak로 백업됨)")
    except Exception as e:
        print(f"오류 발생: {e}") 