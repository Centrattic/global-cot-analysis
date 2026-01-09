import re
import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences based on periods, question marks, exclamation marks, and newlines.
    Multiple consecutive newlines are treated as a single separator.
    
    Args:
        text: The text to split into sentences
        
    Returns:
        List of sentences (non-empty strings)
    """
    if not text:
        return []

    # Normalize newlines: replace one or more newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    sentences = []
    current_sentence = ""

    i = 0
    while i < len(text):
        char = text[i]

        # Check for sentence-ending punctuation
        if char in '.!?':
            current_sentence += char
            # Check if this is followed by whitespace, newline, or end of string
            if i + 1 >= len(text) or text[i + 1] in ' \n':
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
                # Skip the whitespace after punctuation
                if i + 1 < len(text) and text[i + 1] == ' ':
                    i += 1
            i += 1
        # Check for newline (sentence separator)
        elif char == '\n':
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
            i += 1
        else:
            current_sentence += char
            i += 1

    # Add any remaining text as a sentence
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # Filter out empty sentences
    return [s for s in sentences if s]


def chunk(text: str,
          enable_llm_merge: bool = False,
          openrouter_api_key: str = None) -> list[str]:
    """Split text into meaningful chunks per the specified rules.

    Rules summary:
    1) Sentence-level split on ., ?, !, ellipsis ("..." or "…"), and newline; keep terminator; merge multi-terminators.
    2) Merge consecutive math-heavy sentences/lines (< 2 alphabetic words) into one chunk.
    3) Clause-level split on commas, semicolons, colons, and '=>' when the left-side chunk
       (including the delimiter) has >= 2 alphabetic words and >= 2 spaces; keep
       delimiter; avoid numeric delimiters.
    4) Parenthetical split: extract parentheticals with >= 2 spaces inside and at least one non-math-function word.
    5) Post-processing: if a non-math lead-in chunk ends with ':', merge it with the
       immediately following math-heavy chunk.
    """

    def chunk_checker(text: str,
                      chunks: list[str]) -> tuple[bool, int, int, int]:
        char_text = len(text.replace(" ", "").replace("\n", ""))
        char_chunks = sum(len(c.replace(" ", "")) for c in chunks)
        return char_text == char_chunks, char_text, char_chunks, len(chunks)

    def chunk_alphabetic_words(s: str) -> list[str]:
        # Alphabetic word: letters (>=2) possibly joined by apostrophes/quotes or dashes; no digits
        # Allowed connectors: ' ’ " - – —
        return re.findall(r'[A-Za-z]{2,}(?:[\'’"\-–—][A-Za-z]{2,})*', s)

    def count_alpha_words(s: str) -> int:
        return len(chunk_alphabetic_words(s))

    def is_math_heavy(s: str) -> bool:
        return count_alpha_words(s) < 1

    def normalize_whitespace(s: str) -> str:
        return " ".join(s.split()).strip()

    def is_textual(s: str) -> bool:
        return count_alpha_words(s) >= 3 and s.count(" ") >= 2

    def is_textual_right(s: str) -> bool:
        # Lowered threshold for the right side
        return count_alpha_words(s) >= 2 and s.count(" ") >= 2

    def is_ie_eg_abbrev_dot(text_: str, idx: int) -> bool:
        """Return True if the dot at idx is part of 'i.e.'/'i. e.' or 'e.g.'/'e. g.' (case-insensitive).

        Rules:
        - First dot inside the abbreviation is ALWAYS non-terminating (do not split).
        - Second dot is non-terminating only if it is immediately followed (after spaces) by a comma,
          otherwise allow it to behave as a normal terminator (eligible to end a chunk).
        """
        n = len(text_)
        if idx < 0 or idx >= n or text_[idx] != '.':
            return False
        # helper to find previous/next non-space letter index
        def prev_letter(i: int) -> int:
            j = i
            while j >= 0 and text_[j].isspace():
                j -= 1
            return j if j >= 0 and text_[j].isalpha() else -1

        def next_letter(i: int) -> int:
            j = i
            while j < n and text_[j].isspace():
                j += 1
            return j if j < n and text_[j].isalpha() else -1

        pl = prev_letter(idx - 1)
        nl = next_letter(idx + 1)
        # First dot cases: 'i . e .' or 'e . g .'
        if pl != -1 and nl != -1:
            left = text_[pl].lower()
            right = text_[nl].lower()
            # expect trailing dot after the right letter
            nr = nl + 1
            while nr < n and text_[nr].isspace():
                nr += 1
            if nr < n and text_[nr] == '.':
                if left == 'i' and right == 'e':
                    return True
                if left == 'e' and right == 'g':
                    return True
        # Second dot cases: current '.' after 'e' or 'g' where before that is '.' and 'i'/'e'
        # Find the immediate previous letter before this dot
        pl2 = prev_letter(idx - 1)
        if pl2 != -1:
            mid = text_[pl2].lower()
            # find a previous dot before that
            j = pl2 - 1
            while j >= 0 and text_[j].isspace():
                j -= 1
            if j >= 0 and text_[j] == '.':
                # find the letter before that dot
                pl3 = prev_letter(j - 1)
                if pl3 != -1:
                    first = text_[pl3].lower()
                    if (first == 'i' and mid == 'e') or (first == 'e'
                                                         and mid == 'g'):
                        # only block if next non-space char is a comma; otherwise allow terminator
                        k = idx + 1
                        while k < n and text_[k].isspace():
                            k += 1
                        if k < n and text_[k] == ',':
                            return True
        return False

    def split_sentences(s: str) -> list[str]:
        # Scan characters; treat ellipsis and multi-terminators as single boundary; keep terminators
        pieces: list[str] = []
        buf_chars: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            # Newline as terminator
            if ch == "\n":
                buf_chars.append(ch)
                # absorb immediate closing parenthesis after newline
                next_i = i + 1
                if next_i < n and s[next_i] == ")":
                    buf_chars.append(")")
                    i = next_i + 1
                else:
                    i = next_i
                pieces.append("".join(buf_chars))
                buf_chars = []
                continue

            # Ellipsis (unicode) or three dots
            if ch == "…" or (ch == "." and i + 2 < n and s[i + 1] == "."
                             and s[i + 2] == "."):
                if ch == "…":
                    buf_chars.append("…")
                    i += 1
                else:
                    buf_chars.append("...")
                    i += 3
                # absorb trailing ? or ! (multi-terminators)
                while i < n and s[i] in "?!":
                    buf_chars.append(s[i])
                    i += 1
                # absorb immediate closing parenthesis and quotes after terminator
                while i < n and s[i] in {")", '"', "'"}:
                    buf_chars.append(s[i])
                    i += 1
                pieces.append("".join(buf_chars))
                buf_chars = []
                continue

            # Single ., ?, ! terminators (merge runs of ?!)
            if ch in ".?!":
                # Do not treat '.' as terminator when it's a decimal point between digits
                if ch == "." and i - 1 >= 0 and i + 1 < n and s[
                        i - 1].isdigit() and s[i + 1].isdigit():
                    buf_chars.append(ch)
                    i += 1
                    continue
                # Do not treat '.' in common abbreviations i.e./e.g. (with optional spaces) as terminators
                if ch == "." and is_ie_eg_abbrev_dot(s, i):
                    buf_chars.append(ch)
                    i += 1
                    continue
                buf_chars.append(ch)
                i += 1
                # merge consecutive ? or ! (not additional dots)
                while i < n and s[i] in "?!":
                    buf_chars.append(s[i])
                    i += 1
                # absorb immediate closing parenthesis and quotes after terminator
                while i < n and s[i] in {")", '"', "'"}:
                    buf_chars.append(s[i])
                    i += 1
                pieces.append("".join(buf_chars))
                buf_chars = []
                continue

            # Regular character
            buf_chars.append(ch)
            i += 1

        if buf_chars:
            pieces.append("".join(buf_chars))
        return pieces

    def merge_math_runs(sentences: list[str]) -> list[str]:
        merged: list[str] = []
        current_run: list[str] = []
        in_math_run = False
        for sent in sentences:
            if is_math_heavy(sent):
                if not in_math_run:
                    current_run = [sent]
                    in_math_run = True
                else:
                    current_run.append(sent)
            else:
                if in_math_run:
                    merged.append(" ".join(current_run))
                    current_run = []
                    in_math_run = False
                merged.append(sent)
        if in_math_run and current_run:
            merged.append(" ".join(current_run))
        return merged

    def is_numeric_delimiter(text_: str, idx: int) -> bool:
        # idx is position of delimiter , ; :
        j = idx - 1
        while j >= 0 and text_[j].isspace():
            j -= 1
        k = idx + 1
        m = len(text_)
        while k < m and text_[k].isspace():
            k += 1
        left = text_[j] if 0 <= j < m else ""
        right = text_[k] if 0 <= k < m else ""
        return left.isdigit() and right.isdigit()

    def split_clauses(piece: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        i = 0
        while i < len(piece):
            ch = piece[i]
            # Handle two-char operator '=>'
            if ch == "=" and i + 1 < len(piece) and piece[i + 1] == ">":
                # keep operator with left side
                buf.append("=>")
                candidate_left = "".join(buf)
                candidate_right = piece[i + 2:]
                if is_textual(candidate_left) and is_textual_right(
                        candidate_right):
                    parts.append(candidate_left)
                    buf = []
                # move past operator
                i += 2
                continue
            if ch in ",;:":
                if is_numeric_delimiter(piece, i):
                    buf.append(ch)
                    i += 1
                    continue
                # keep delimiter with left side
                buf.append(ch)
                candidate_left = "".join(buf)
                candidate_right = piece[i + 1:]
                if is_textual(candidate_left) and is_textual_right(
                        candidate_right):
                    parts.append(candidate_left)
                    buf = []
                i += 1
                # continue without skipping spaces; they will stick to next clause
                continue
            buf.append(ch)
            i += 1
        if buf:
            parts.append("".join(buf))
        return parts

    def parenthetical_extract(piece: str) -> list[str]:
        # Extract qualifying parentheticals into their own chunks.
        math_funcs = {
            "value", "~", "sin", "cos", "tan", "log", "ln", "exp", "sqrt"
        }
        out: list[str] = []
        emit_from = 0  # start of the next segment to emit
        scan_pos = 0  # current regex scan position
        found_any = False
        while True:
            match = re.search(r"\([^()]*\)", piece[scan_pos:])
            if not match:
                break
            start = scan_pos + match.start()
            end = scan_pos + match.end()
            paren = piece[start:end]
            inside = paren[1:-1]
            has_two_spaces = inside.count(" ") >= 2
            words = chunk_alphabetic_words(inside)
            has_non_math_word = any(w.lower() not in math_funcs for w in words)
            # Require at least 2 alphabetic words to avoid extracting math-y parentheses like (1*b + 7)
            if has_two_spaces and has_non_math_word and len(words) >= 1:
                found_any = True
                # absorb trailing punctuation immediately after ')'
                if end < len(piece) and piece[end] in ".,;:!?":
                    paren = paren + piece[end]
                    end += 1
                # prefix-merge: include adjacent token before '(' if no whitespace
                pre = piece[emit_from:start]
                if pre and not pre[-1].isspace():
                    j = len(pre) - 1
                    while j >= 0 and not pre[j].isspace():
                        j -= 1
                    x = pre[j + 1:]
                    pre = pre[:j + 1]
                    paren = x + paren
                if pre:
                    out.append(pre)
                out.append(paren)
                emit_from = end
                scan_pos = end
            else:
                # Not qualifying; do not split, but continue scanning after this parenthetical
                # Preserve content by not advancing emit_from; just continue scan
                scan_pos = end
        if not found_any:
            return [piece]
        if emit_from < len(piece):
            out.append(piece[emit_from:])
        return out

    if not text:
        return [], chunk_checker(text, [])

    # 1) Sentence-level split
    sentences = split_sentences(text)
    # 2) Merge math-heavy runs
    primary_chunks = merge_math_runs(sentences)

    # 3) Clause-level split per chunk
    clause_chunks: list[str] = []
    for ch in primary_chunks:
        clause_chunks.extend(split_clauses(ch))

    # 4) Parenthetical split
    final_chunks: list[str] = []
    for ch in clause_chunks:
        final_chunks.extend(parenthetical_extract(ch))

    # 5) Post-processing merge: lead-in ':' + following math-heavy chunk
    merged_chunks: list[str] = []
    i = 0
    while i < len(final_chunks):
        current = final_chunks[i]
        if current.rstrip().endswith(":") and not is_math_heavy(current) and (
                i + 1) < len(final_chunks):
            nxt = final_chunks[i + 1]
            if is_math_heavy(nxt):
                merged_chunks.append(f"{current} {nxt}")
                i += 2
                continue
        merged_chunks.append(current)
        i += 1

    # 6) Post-processing merge: if previous ends with ellipsis and next starts with a comma, merge
    punct_merged_chunks: list[str] = []
    i = 0
    while i < len(merged_chunks):
        current = merged_chunks[i]
        j = i
        # try to chain merges with subsequent neighbors as long as rules apply
        while j + 1 < len(merged_chunks):
            nxt = merged_chunks[j + 1]
            cur_trim = current.rstrip()
            # Rule A: ellipsis followed by comma-start
            merged = False
            if (cur_trim.endswith("…") or cur_trim.endswith("...")):
                k = 0
                nlen = len(nxt)
                while k < nlen and nxt[k].isspace():
                    k += 1
                if k < nlen and nxt[k] == ',':
                    current = current + nxt
                    j += 1
                    merged = True
                    continue
            # Rule B: equals sign lead-in followed by math-y start
            if cur_trim.endswith('='):
                k = 0
                nlen = len(nxt)
                while k < nlen and nxt[k].isspace():
                    k += 1
                if k < nlen:
                    ch = nxt[k]
                    if ch == '(' or ch.isdigit() or ch in "+-*/^=%":
                        current = current + nxt
                        j += 1
                        merged = True
                        continue
            # Rule C: parenthetical term followed by division starts of a fraction
            if cur_trim.endswith(')'):
                k = 0
                nlen = len(nxt)
                while k < nlen and nxt[k].isspace():
                    k += 1
                if k < nlen and nxt[k] == '/':
                    current = current + nxt
                    j += 1
                    merged = True
                    continue
            # if none merged, break
            if not merged:
                break
        punct_merged_chunks.append(current)
        i = j + 1

    # 7) Post-processing merge: coalesce consecutive math-heavy chunks
    coalesced_chunks: list[str] = []
    math_buf: list[str] = []
    for c in punct_merged_chunks:
        if is_math_heavy(c):
            math_buf.append(c)
        else:
            if math_buf:
                coalesced_chunks.append(" ".join(math_buf))
                math_buf = []
            coalesced_chunks.append(c)
    if math_buf:
        coalesced_chunks.append(" ".join(math_buf))

    # Normalize whitespace and drop empties
    normalized = [normalize_whitespace(c) for c in coalesced_chunks]
    chunks_only = [c for c in normalized if c]

    # Enforce numbering boundary rules:
    # - A chunk must not end with ': 1.'/': 2.'/': 3.'; move the numbering to the next chunk.
    # - A chunk that is exactly '{number}.' must be merged with the following chunk (if any).
    chunks_only = _apply_colon_number_rules(chunks_only)

    if enable_llm_merge:
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when enable_llm_merge=True")
        chunks_only = _llm_merge_chunks(chunks_only, api_key)

    return chunks_only, chunk_checker(text, chunks_only)


def _build_merge_prompt(prev2: str, prev1: str, a: str, b: str, next1: str,
                        next2: str) -> list[dict]:
    system = (
        "You judge if two consecutive chunks should be merged into one. Output exactly YES or NO (uppercase), with no other text."
        "Guidelines:\n"
        "YES if A and B are parts of the same math equation or expression.\n"
        "YES if B continues the phrase or idea begun in A; or if a unit like 'i.e.' or 'e.g.' is split between A and B.\n"
        "YES if A and B go together.\n\n"
        "NO if: A is a sentence, or if A ends with '=>', or if B starts with a capital letter.\n\n"
    )
    user = (
        "Previous context:\n"
        "Prev-2:\n<BEGIN_PREV2>\n" + prev2 + "\n<END_PREV2>\n\n"
        "Prev-1:\n<BEGIN_PREV1>\n" + prev1 + "\n<END_PREV1>\n\n"
        "Chunk A:\n<BEGIN_A>\n" + a + "\n<END_A>\n\n"
        "Chunk B:\n<BEGIN_B>\n" + b + "\n<END_B>\n\n"
        "Following context:\n"
        "Next+1:\n<BEGIN_NEXT1>\n" + next1 + "\n<END_NEXT1>\n\n"
        "Next+2:\n<BEGIN_NEXT2>\n" + next2 + "\n<END_NEXT2>\n\n"
        "Output: YES or NO."
        "YES if A and B are parts of the same math equation or expression.\n"
        "YES if B continues the phrase or idea begun in A; or if a unit like 'i.e.' or 'e.g.' is split between A and B.\n"
        "YES if A and B go together.\n\n")
    return [{
        "role": "system",
        "content": system
    }, {
        "role": "user",
        "content": user
    }]


def _judge_pairs_parallel(chunks: list[str], api_key: str) -> list[bool]:
    if len(chunks) < 2:
        return []
    base_url = "https://openrouter.ai/api/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    total_pairs = len(chunks) - 1

    # Only consider pairs where at least one chunk is short (<= 3 alphabetic words)
    def count_alpha_words_local(s: str) -> int:
        return len(re.findall(r"[A-Za-z]{2,}(?:['’\-–—][A-Za-z]{2,})*", s))

    def ends_with_terminator(s: str) -> bool:
        t = s.rstrip()
        return bool(t) and t[-1] in ".?!"

    def starts_with_capital(s: str) -> bool:
        j = 0
        n = len(s)
        while j < n and s[j].isspace():
            j += 1
        return j < n and s[j].isalpha() and s[j].isupper()

    indices = []
    for i in range(total_pairs):
        a = chunks[i]
        b = chunks[i + 1]
        if not (count_alpha_words_local(a) <= 3
                or count_alpha_words_local(b) <= 3):
            continue
        if ends_with_terminator(a) and starts_with_capital(b):
            continue
        indices.append(i)
    print(f"LLM pairs considered: {len(indices)} of {total_pairs}")

    def task(i: int) -> tuple[int, bool]:
        prev2 = chunks[i - 2] if i - 2 >= 0 else ""
        prev1 = chunks[i - 1] if i - 1 >= 0 else ""
        a = chunks[i]
        b = chunks[i + 1]
        next1 = chunks[i + 2] if i + 2 < len(chunks) else ""
        next2 = chunks[i + 3] if i + 3 < len(chunks) else ""
        messages = _build_merge_prompt(prev2, prev1, a, b, next1, next2)
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages,
            "max_tokens": 2,
            "temperature": 0.3,
            "provider": {
                "only": ["openai"]
            },
        }
        r = requests.post(f"{base_url}/chat/completions",
                          headers=headers,
                          data=json.dumps(payload))
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip().upper()
        return i, (content == "YES")

    # Avoid spawning too many threads (this function may run inside other thread pools)
    try:
        max_workers_cap = int(os.getenv("CHUNK_LLM_MAX_WORKERS", "300"))
    except Exception:
        max_workers_cap = 4
    max_workers = min(max_workers_cap, len(indices)) if indices else 1
    results: list[bool] = [False] * total_pairs
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if indices:
            futures = {ex.submit(task, i): i for i in indices}
            for fut in as_completed(futures):
                idx, decision = fut.result()
                results[idx] = decision
    return results


def _merge_by_yes_runs(chunks: list[str], yes_pairs: list[bool]) -> list[str]:
    if not yes_pairs:
        return chunks
    n = len(chunks)
    out: list[str] = []
    i = 0
    while i < n:
        if i < n - 1 and yes_pairs[i]:
            j = i
            while j < n - 1 and yes_pairs[j]:
                j += 1
            merged = " ".join(chunks[i:j + 1])
            out.append(merged)
            i = j + 1
        else:
            out.append(chunks[i])
            i += 1
    return out


def _llm_merge_chunks(consecutive_chunks: list[str],
                      api_key: str) -> list[str]:
    decisions = _judge_pairs_parallel(consecutive_chunks, api_key)
    return _merge_by_yes_runs(consecutive_chunks, decisions)


def _apply_colon_number_rules(chunks: list[str]) -> list[str]:
    """Post-process chunks to enforce numbering boundary rules.

    - If a chunk is exactly '{number}.' or a single-letter like 'A.', merge it with the following chunk
      (unless it is the last chunk).
    """
    if not chunks:
        return chunks

    # Merge standalone '{number}.' chunks with the following chunk
    out: list[str] = []
    j = 0
    while j < len(chunks):
        cur = chunks[j]
        if re.fullmatch(r"(?:\d+|[A-Za-z])\.", cur):
            if j + 1 < len(chunks):
                merged = f"{cur} {chunks[j + 1].lstrip()}"
                out.append(merged)
                j += 2
                continue
        out.append(cur)
        j += 1

    return out
