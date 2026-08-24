"""
High-Performance Stopping Criteria for Text Generation in NAA
"""

from typing import List, Any, Optional
from src.core.prompt import DEFAULT_STOP_TOKENS

def get_combined_stop_tokens(custom_stops: Optional[List[str]] = None) -> List[str]:
    """
    Returns the consolidated list of stop tokens passed directly to llama.cpp or transformers engine.
    """
    stops = list(DEFAULT_STOP_TOKENS)
    if custom_stops:
        for s in custom_stops:
            if s and s not in stops:
                stops.append(s)
    return stops

class WindowedStringStopCriteria:
    """
    Windowed stopping criteria for fallback/transformers tokenizers.
    Inspects trailing window of generated tokens.
    """
    def __init__(self, tok_inst: Any, stop_words: List[str], input_length: int, max_window_tokens: int = 16):
        self.tok_inst = tok_inst
        self.stop_words = stop_words
        self.input_length = input_length
        self.max_window_tokens = max_window_tokens

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs) -> bool:
        if not input_ids or len(input_ids) == 0:
            return False
        first_seq = input_ids[0]
        gen_ids = first_seq[self.input_length:]
        if len(gen_ids) == 0:
            return False
        window_ids = gen_ids[-self.max_window_tokens:]
        window_text = self.tok_inst.decode(window_ids, skip_special_tokens=False)
        for sw in self.stop_words:
            if sw in window_text:
                return True
        return False
