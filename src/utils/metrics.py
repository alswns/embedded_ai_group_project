"""
평가 메트릭 함수
"""
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    print("⚠️ nltk가 설치되지 않았습니다. METEOR 점수 계산 불가.")
    METEOR_AVAILABLE = False
    meteor_score = None
    word_tokenize = None

def calculate_meteor(generated_caption, reference_caption):
    """METEOR 점수 계산
    
    Args:
        generated_caption: 생성된 캡션 (리스트 또는 문자열)
        reference_caption: 참조 캡션 (문자열)
    
    Returns:
        METEOR 점수 (float) 또는 None
    """
    if not METEOR_AVAILABLE:
        return None
    
    try:
        # 생성된 캡션이 리스트인 경우 처리
        if isinstance(generated_caption, list):
            # 특수 토큰 제거
            gen_words = [w for w in generated_caption if w not in ['<start>', '<end>', '<pad>', '<unk>']]
            gen_words_str = ' '.join(gen_words)
        else:
            gen_words_str = str(generated_caption)
        
        if not gen_words_str:
            return None
        
        # 토큰화
        ref_words = word_tokenize(reference_caption.lower())
        gen_tokens = word_tokenize(gen_words_str.lower())
        
        if not gen_tokens:
            return None
        
        score = meteor_score([ref_words], gen_tokens)
        return score
    except Exception as e:
        return None

