from typing import Dict, Any, List
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import re

from src.utils.file_utils import FileUtils

LABELS = {
    "Q": "question",
    "A": "answer",
    "N": "neither",
}


def build_prompt(prompt_text: str, cluster_texts: List[str]) -> str:
    head = (
        "You will receive a problem P and a cluster X of sentences. "
        "Output exactly one character: Q, A, or N.\n"
        "Q: only if X is a restatement of the problem or of part of the problem P.\n"
        "A: only if X is a an EXPLICIT answer to the EXACT question posed in problem P.\n"
        "N: in all other cases.\n"
        "No explanations. Only Q or A or N.\n\n"
    )
    cluster_block = "\n".join(f"- {t}" for t in cluster_texts[:20])  # cap for safety
    return f"{head}P:\n{prompt_text}\n\nX (cluster sentences, examples):\n{cluster_block}\n\nAnswer (Q/A/N):"


def call_llm(
    prompt: str, model: str | None = None, max_tokens: int | None = None, raw: bool = False
) -> str:
    # Use same provider as cluster merging (OpenRouter chat/completions)
    import requests

    # Load environment variables from .env if present
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    base_url = "https://openrouter.ai/api/v1"
    model = model or os.getenv("CLUSTER_LLM_MODEL", "openai/gpt-4o-mini")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if model.startswith("openai/"):
        payload["provider"] = {"only": ["openai"]}
    elif model.startswith("google/"):
        # Prefer Google Vertex for Gemini 2.5 via OpenRouter
        if model.startswith("google/gemini-2.5"):
            payload["provider"] = {"only": ["google-vertex"], "allow_fallbacks": False}
        else:
            payload["provider"] = {"only": ["google"]}
    max_retries = int(os.environ.get("LABELER_MAX_RETRIES", "3"))
    delay = float(os.environ.get("LABELER_RETRY_DELAY", "0.6"))
    content = None
    for attempt in range(max_retries + 1):
        resp = requests.post(
            f"{base_url}/chat/completions", headers=headers, data=json.dumps(payload)
        )
        status = resp.status_code
        if status != 200:
            if attempt < max_retries:
                time.sleep(delay)
                continue
            raise RuntimeError(f"Labeler HTTP {status}: {resp.text}")
        data = resp.json()
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                break
        if attempt < max_retries:
            time.sleep(delay)
            continue
        raise RuntimeError(f"Labeler invalid response: {data}")
    if raw:
        return content.strip()
    text = content.strip().upper()[:1]
    if text not in ("Q", "A", "N"):
        return "N"
    return text


def extract_cluster_sentences(node_data: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    s_list = node_data.get("sentences", [])
    for s in s_list:
        t = s.get("text")
        if t:
            out.append(t)
    rep = node_data.get("representative_sentence")
    if rep:
        out.append(rep)
    # dedupe preserving order
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def label_flowchart(
    flowchart: Dict[str, Any], prompt_text: str, max_workers: int | None = None
) -> Dict[str, Any]:
    # flowchart["nodes"] is a list of {cluster_key: node_data}
    nodes = [n for n in flowchart.get("nodes", []) if isinstance(n, dict) and n]
    work: List[tuple[str, Dict[str, Any], str, List[str]]] = []
    for node_obj in nodes:
        cluster_key = next(iter(node_obj.keys()))
        node_data = node_obj[cluster_key]
        texts = extract_cluster_sentences(node_data)
        prompt = build_prompt(prompt_text, texts)
        work.append((cluster_key, node_data, prompt, texts))

    if max_workers is None:
        env_val = os.environ.get("LABELER_MAX_WORKERS")
        max_workers = int(env_val) if env_val and env_val.isdigit() else 8

    def run_prompt(item: tuple[str, Dict[str, Any], str, List[str]]) -> tuple[str, str]:
        ck, _, pr, texts = item
        primary_model = os.environ.get("CLUSTER_LABEL_PRIMARY_MODEL", "openai/gpt-4o-mini")
        secondary_model = os.environ.get("CLUSTER_LABEL_SECOND_MODEL", "google/gemini-2.5-flash")
        s1 = call_llm(pr, model=primary_model, max_tokens=1)
        if s1 == "Q":
            s2 = call_llm(pr, model=secondary_model, max_tokens=1)
            return ck, LABELS.get(s2, "neither")
        if s1 == "A":
            mentions = 0
            for t in texts:
                tt = t.lower()
                if ("answer" in tt) or ("output" in tt):
                    mentions += 1
            if mentions >= 2:
                s2 = call_llm(pr, model=primary_model, max_tokens=1)
                if s2 == "A":
                    # Final step: extract numeric answer from representative sentence
                    representative = texts[0] if texts else ""
                    extract_prompt = (
                        "S contains a number. Extract it.\n"
                        f"S: {representative}\n"
                        "Only output the number from S."
                    )
                    out = call_llm(extract_prompt, model=secondary_model, raw=True)
                    m = re.search(r"[+-]?\d+(?:\.\d+)?", out.strip())
                    if m:
                        return ck, m.group(0)
                    return ck, LABELS["N"]
                return ck, LABELS["N"]
            return ck, LABELS["N"]
        return ck, LABELS.get(s1, "neither")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_prompt, it): it for it in work}
        for fut in as_completed(futures):
            ck, label = fut.result()
            obj = next(o for o in nodes if next(iter(o.keys())) == ck)
            obj[ck]["label"] = label
    # Post-process: mark self-checking clusters with stricter criteria
    for obj in nodes:
        ck = next(iter(obj.keys()))
        nd = obj[ck]
        lbl = nd.get("label")
        if lbl == "neither":
            # Count chunks (with counts) that contain 'check' or 'verify'
            total_chunks = 0
            hits = 0
            # Representative sentence counts as one chunk
            rep = nd.get("representative_sentence")
            if isinstance(rep, str):
                total_chunks += 1
                low = rep.lower()
                if ("check" in low) or ("verify" in low):
                    hits += 1
            # Count sentences with their frequency
            for s in nd.get("sentences", []) or []:
                if not isinstance(s, dict):
                    continue
                t = s.get("text")
                c = s.get("count", 1)
                if not isinstance(c, int):
                    c = 1
                if isinstance(t, str):
                    total_chunks += c
                    low = t.lower()
                    if ("check" in low) or ("verify" in low):
                        hits += c
            if total_chunks <= 1:
                if hits >= 1:
                    nd["label"] = "self-checking"
            else:
                if hits >= 2:
                    nd["label"] = "self-checking"
    return flowchart


def load_prompt_text(prompts_json_path: str, prompt_id: str) -> str:
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        p = json.load(f)
    entry = p.get(prompt_id, {})
    if isinstance(entry, dict):
        return entry.get("text", "")
    if isinstance(entry, str):
        return entry
    return ""


def label_flowchart_file(
    flowchart_path: str,
    prompts_json_path: str = FileUtils.get_prompts_file_path(),
    max_workers: int | None = None,
) -> None:
    with open(flowchart_path, "r", encoding="utf-8") as f:
        flowchart = json.load(f)
    prompt_id = flowchart.get("prompt_index", "")
    prompt_text = load_prompt_text(prompts_json_path, prompt_id)
    labeled = label_flowchart(flowchart, prompt_text, max_workers=max_workers)
    with open(flowchart_path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)


__all__ = [
    "label_flowchart",
    "load_prompt_text",
    "label_flowchart_file",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add cluster labels to an existing flowchart JSON")
    parser.add_argument("flowchart", type=str, help="Path to flowchart JSON")
    parser.add_argument(
        "--prompts", type=str, default=FileUtils.get_prompts_file_path(), help="Path to prompts.json"
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Parallel workers for labeling",
    )
    args = parser.parse_args()

    label_flowchart_file(
        args.flowchart, prompts_json_path=args.prompts, max_workers=args.max_workers
    )
    print(f"Labeled flowchart written back to {args.flowchart}")
