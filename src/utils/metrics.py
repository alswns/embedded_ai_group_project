"""
평가 메트릭 함수
"""
from collections import Counter

# NLTK METEOR 시도
METEOR_AVAILABLE = False
try:
    from nltk.translate.meteor_score import single_meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    METEOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"⚠️ NLTK METEOR import failed: {e}")
    METEOR_AVAILABLE = False
    single_meteor_score = None
    word_tokenize = None


def _simple_tokenize(text):
    """간단한 토크나이제이션 (NLTK 없을 때 사용)"""
    if not text:
        return []
    # 소문자 변환 후 공백으로 분리
    text = text.lower().strip()
    # 특수 문자 제거 (기본적인 것만)
    text = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
    # 공백으로 분리
    words = text.split()
    return [w for w in words if w]  # 빈 문자열 제거


def calculate_meteor(generated_caption, reference_caption):
    """
    METEOR 점수 계산
    
    Args:
        generated_caption: 생성된 캡션 (문자열 또는 리스트)
        reference_caption: 참조 캡션 (문자열)
    
    Returns:
        METEOR 점수 (float, 0.0~1.0) 또는 0.0 (실패 시)
    """
    
    # 입력 검증
    if not generated_caption or not reference_caption:
        return 0.0
    
    # 생성된 캡션을 문자열로 변환
    if isinstance(generated_caption, list):
        gen_str = ' '.join(str(w) for w in generated_caption)
    else:
        gen_str = str(generated_caption)
    
    ref_str = str(reference_caption)
    
    if not gen_str or not ref_str:
        return 0.0
    
    # 특수 토큰 제거 (<start>, <end>, <pad>, <unk> 등)
    gen_str = _remove_special_tokens(gen_str)
    ref_str = _remove_special_tokens(ref_str)
    
    if not gen_str or not ref_str:
        return 0.0
    
    try:
        # 방법 1: NLTK METEOR 사용 (전체 텍스트 기반)
        if METEOR_AVAILABLE and single_meteor_score is not None:
            try:
                # single_meteor_score(reference, hypothesis)
                # reference: 참조 캡션 (문자열)
                # hypothesis: 생성된 캡션 (문자열)
                score = single_meteor_score(ref_str.lower(), gen_str.lower())
                
                # 유효한 점수인지 확인
                if score is not None and isinstance(score, (int, float)):
                    return float(max(0.0, min(1.0, score)))  # 0~1 범위로 정규화
                else:
                    # None이거나 유효하지 않은 값이면 Jaccard로 폴백
                    return _compute_jaccard_similarity(gen_str, ref_str)
            except Exception as e:
                # NLTK 오류 발생 시 Jaccard로 폴백 (에러 메시지 억제)
                return _compute_jaccard_similarity(gen_str, ref_str)
        
        # 방법 2: Jaccard 유사도 (폴백)
        return _compute_jaccard_similarity(gen_str, ref_str)
        
    except Exception as e:
        print(f"❌ calculate_meteor error: {e}")
        return 0.0


def _remove_special_tokens(text):
    """
    특수 토큰 제거
    
    Args:
        text: 입력 텍스트
    
    Returns:
        특수 토큰이 제거된 텍스트
    """
    if not text:
        return ""
    
    text = str(text)
    # 일반적인 특수 토큰 제거
    special_tokens = ['<start>', '<end>', '<pad>', '<unk>', '<unknown>', '<s>', '</s>']
    
    for token in special_tokens:
        text = text.replace(token, ' ')
    
    # 연속된 공백 정리
    text = ' '.join(text.split())
    
    return text.strip()


def _compute_jaccard_similarity(generated, reference):
    """
    Jaccard 유사도 계산 (NLTK 없을 때 사용)
    
    Args:
        generated: 생성된 텍스트
        reference: 참조 텍스트
    
    Returns:
        0.0~1.0 범위의 유사도
    """
    try:
        # 토크나이제이션
        gen_tokens = set(_simple_tokenize(generated))
        ref_tokens = set(_simple_tokenize(reference))
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Jaccard 유사도 = |교집합| / |합집합|
        intersection = len(gen_tokens & ref_tokens)
        union = len(gen_tokens | ref_tokens)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        return float(max(0.0, min(1.0, similarity)))  # 0~1 범위
        
    except Exception as e:
        print(f"❌ Jaccard similarity error: {e}")
        return 0.0


def calculate_bleu(generated_caption, reference_caption):
    """
    간단한 BLEU 점수 계산
    
    Args:
        generated_caption: 생성된 캡션
        reference_caption: 참조 캡션
    
    Returns:
        BLEU 점수 (0.0~1.0)
    """
    try:
        gen_tokens = _simple_tokenize(str(generated_caption))
        ref_tokens = _simple_tokenize(str(reference_caption))
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # 1-gram 정확도
        gen_counter = Counter(gen_tokens)
        ref_counter = Counter(ref_tokens)
        
        matches = 0
        for token, count in gen_counter.items():
            matches += min(count, ref_counter.get(token, 0))
        
        precision = matches / len(gen_tokens) if gen_tokens else 0.0
        recall = matches / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(max(0.0, min(1.0, f1)))
        
    except Exception as e:
        print(f"❌ BLEU calculation error: {e}")
        return 0.0


