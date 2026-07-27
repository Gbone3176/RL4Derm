import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def _content_blocks_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                texts.append(str(part.get("text") or part.get("content") or ""))
            else:
                texts.append(str(part))
        return "\n".join(text for text in texts if text)
    return str(content)

def _message_to_text(value):
    if isinstance(value, dict):
        return _content_blocks_to_text(value.get("content", value.get("text", "")))
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return _message_to_text(item)
        return "\n".join(_message_to_text(item) for item in value)
    return "" if value is None else str(value)

def _extract_answer_text(text):
    text = _message_to_text(text).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def _extract_tagged_answer_text(text):
    text = _message_to_text(text).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def _normalize_text_answer(text):
    text = _extract_answer_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.strip()

def _option_letter(text):
    match = re.match(r"\s*([a-z])\s*[\).:-]?", _normalize_text_answer(text), re.IGNORECASE)
    return match.group(1).lower() if match else None

def _gold_mcqa_parts(text):
    raw = _message_to_text(text).strip()
    match = re.match(r"\s*([A-Za-z])\s*[\).:-]?\s*(.*?)\s*$", raw)
    if not match:
        return None, _normalize_plain_text(raw)
    return match.group(1).upper(), _normalize_plain_text(match.group(2))

def _normalize_plain_text(text):
    text = _message_to_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text.strip()

def _mcqa_accuracy(answer_text, gold_text):
    if answer_text is None:
        return 0.0
    gold_letter, gold_choice = _gold_mcqa_parts(gold_text)
    if gold_letter is None:
        return 0.0

    pred_raw = _message_to_text(answer_text).strip()
    pred_norm = _normalize_plain_text(pred_raw)
    pred_letter_match = re.match(r"\s*([A-Za-z])\s*([\).:-]\s*)?(.*?)\s*$", pred_raw)
    pred_letter = pred_letter_match.group(1).upper() if pred_letter_match else None
    pred_tail = _normalize_plain_text(pred_letter_match.group(3)) if pred_letter_match else ""

    if pred_raw.upper() == gold_letter:
        return 1.0
    if pred_letter == gold_letter:
        return 0.5
    if gold_choice and pred_norm == gold_choice:
        return 0.5
    if gold_choice and pred_norm == f"{gold_letter.lower()} {gold_choice}":
        return 0.5
    if gold_choice and pred_tail == gold_choice and pred_letter == gold_letter:
        return 0.5
    return 0.0

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []

    for completion, sol in zip(completions, assistant):
        completion_raw = _message_to_text(completion)
        sol_raw = _message_to_text(sol)
        tagged_answer = _extract_tagged_answer_text(completion_raw)
        gold_letter, gold_choice = _gold_mcqa_parts(sol_raw)
        if gold_letter is not None:
            rewards.append(_mcqa_accuracy(tagged_answer, sol_raw))
            continue
        try:
            gold_parsed = parse(sol_raw, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion_raw,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion_raw}, gold: {sol_raw}")
                reward = None
        else:
            # fallback to text match
            completion_text = _normalize_text_answer(completion_raw)
            gold_text = _normalize_text_answer(sol_raw)
            completion_letter = _option_letter(completion_raw)
            gold_letter = _option_letter(sol_raw)
            reward = float(
                completion_text == gold_text
                or (
                    completion_letter is not None
                    and gold_letter is not None
                    and completion_letter == gold_letter
                )
            )

        rewards.append(reward)

    return rewards

STRICT_FORMAT_PATTERN = re.compile(
    r"^<think>(?P<think>.*?)</think>\s*<answer>\s*(?P<answer>[A-Za-z])\s*</answer>$",
    re.DOTALL,
)


def analyze_format_completion(content):
    """Return strict/fullmatch diagnostics for the required GRPO MCQA format.

    Strict target: the entire completion is only
    ``<think>non-empty reasoning</think><answer>single-letter</answer>``.
    Whitespace between tags is tolerated, but any non-whitespace text outside
    the two tag bodies is diagnosed as ``no_text_outside_tags=False``.
    """
    text = _message_to_text(content).strip()
    tag_counts = {tag: text.count(tag) for tag in ("<think>", "</think>", "<answer>", "</answer>")}
    duplicate_tags = any(count > 1 for count in tag_counts.values())
    missing_tags = any(count == 0 for count in tag_counts.values())
    four_tag_presence = all(count >= 1 for count in tag_counts.values())
    exact_one_each_tag = all(count == 1 for count in tag_counts.values())

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    think_text = think_match.group(1) if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""
    think_nonempty = bool(think_text.strip())
    answer_single_letter = bool(re.fullmatch(r"[A-Za-z]", answer_text))

    no_text_outside_tags = False
    outside_text = text
    if think_match and answer_match and think_match.start() <= think_match.end() <= answer_match.start():
        outside_parts = [
            text[: think_match.start()],
            text[think_match.end() : answer_match.start()],
            text[answer_match.end() :],
        ]
        outside_text = "".join(outside_parts)
        no_text_outside_tags = outside_text.strip() == ""

    empty_think_prefix = bool(re.match(r"^<think>\s*</think>", text, re.DOTALL))
    strict_match = STRICT_FORMAT_PATTERN.fullmatch(text)
    strict_format = bool(
        strict_match
        and exact_one_each_tag
        and think_nonempty
        and answer_single_letter
        and no_text_outside_tags
    )

    diagnostics = {
        "strict_format": strict_format,
        "strict_fullmatch": bool(strict_match),
        "think_nonempty": think_nonempty,
        "no_text_outside_tags": no_text_outside_tags,
        "answer_single_letter": answer_single_letter,
        "empty_think_prefix": empty_think_prefix,
        "four_tag_presence": four_tag_presence,
        "exact_one_each_tag": exact_one_each_tag,
        "duplicate_tags": duplicate_tags,
        "missing_tags": missing_tags,
        "tag_counts": tag_counts,
        "think_chars": len(think_text),
        "think_nonspace_chars": len(re.sub(r"\s+", "", think_text)),
        "answer_text": answer_text,
        "outside_text_chars": len(outside_text),
        "outside_text_nonspace_chars": len(re.sub(r"\s+", "", outside_text)),
    }
    diagnostics["format_reward"] = _format_reward_from_diagnostics(diagnostics)
    return diagnostics


def _format_reward_from_diagnostics(diagnostics):
    if diagnostics["strict_format"]:
        return 1.0

    # Hard reject the failure mode seen in run 13: an empty <think></think>
    # prefix followed by reasoning outside tags must not receive partial credit.
    if diagnostics["empty_think_prefix"]:
        return 0.0
    if not diagnostics["no_text_outside_tags"] and diagnostics["outside_text_nonspace_chars"] > 0:
        return 0.0
    if diagnostics["duplicate_tags"] or diagnostics["missing_tags"]:
        return 0.0

    reward = 0.0
    # Strict safety rails above prevent rewarding malformed outside text; this
    # residual shaping is only for nearly-canonical, tag-contained attempts.
    if diagnostics["exact_one_each_tag"]:
        reward += 0.20
    if diagnostics["think_nonempty"]:
        reward += 0.30
    if diagnostics["answer_single_letter"]:
        reward += 0.30
    if diagnostics["no_text_outside_tags"]:
        reward += 0.15

    return max(0.0, min(0.95, round(reward, 6)))


def format_reward(completions, **kwargs):
    """Reward required MCQA tag format with strict target plus partial shaping."""
    return [analyze_format_completion(content)["format_reward"] for content in completions]
