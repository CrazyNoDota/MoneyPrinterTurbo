import json
import logging
import random
import re
import time
import requests
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config

_max_retries = 5
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    """Exponential backoff with jitter between LLM retries.

    The retry loops below previously re-called the provider immediately, which
    hammers a rate-limited API. Sleeping between attempts lets a transient 429 /
    5xx recover instead of burning every attempt in a few milliseconds.
    """
    delay = min(base * (2 ** attempt), cap)
    delay += random.random() * delay
    logger.info(f"backing off {delay:.1f}s before retry")
    time.sleep(delay)
_DEPRECATED_GEMINI_MODELS = {"gemini-pro", "gemini-1.0-pro"}


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    content = content.strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content.replace("\n", "")


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _generate_response(prompt: str) -> str:
    try:
        content = ""
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        if llm_provider == "g4f":
            if not config.app.get("enable_g4f", False):
                raise ValueError(
                    "g4f provider is disabled by default because it relies on "
                    "reverse-engineered third-party endpoints. Set enable_g4f=true "
                    "in config.toml only if you understand and accept the security, "
                    "reliability, and legal risks."
                )

            logger.warning(
                "g4f provider is enabled. This provider may be unstable and carries "
                "supply-chain and terms-of-service risks. Prefer official providers, "
                "OpenAI-compatible APIs, LiteLLM, Ollama, or local inference for production."
            )
            try:
                import g4f
            except ImportError as e:
                raise ValueError(
                    "g4f package is not installed by default. Install the optional "
                    "dependency with `uv sync --extra g4f` only if you understand "
                    "and accept the provider risks."
                ) from e

            model_name = config.app.get("g4f_model_name", "")
            if not model_name:
                model_name = "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            api_version = ""  # for azure
            if llm_provider == "moonshot":
                api_key = config.app.get("moonshot_api_key")
                model_name = config.app.get("moonshot_model_name")
                base_url = "https://api.moonshot.cn/v1"
            elif llm_provider == "ollama":
                # api_key = config.app.get("openai_api_key")
                api_key = "ollama"  # any string works but you are required to have one
                model_name = config.app.get("ollama_model_name")
                base_url = config.app.get("ollama_base_url", "")
                if not base_url:
                    base_url = "http://localhost:11434/v1"
            elif llm_provider == "openai":
                api_key = config.app.get("openai_api_key")
                model_name = config.app.get("openai_model_name")
                base_url = config.app.get("openai_base_url", "")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
            elif llm_provider == "oneapi":
                api_key = config.app.get("oneapi_api_key")
                model_name = config.app.get("oneapi_model_name")
                base_url = config.app.get("oneapi_base_url", "")
            elif llm_provider == "azure":
                api_key = config.app.get("azure_api_key")
                model_name = config.app.get("azure_model_name")
                base_url = config.app.get("azure_base_url", "")
                api_version = config.app.get("azure_api_version", "2024-02-15-preview")
            elif llm_provider == "gemini":
                api_key = config.app.get("gemini_api_key")
                model_name = config.app.get("gemini_model_name")
                base_url = config.app.get("gemini_base_url", "")
                # Gemini 旧模型名已经陆续下线，这里自动兼容历史配置，
                # 避免用户沿用旧值时直接收到 404。
                if not model_name:
                    model_name = _DEFAULT_GEMINI_MODEL
                elif model_name in _DEPRECATED_GEMINI_MODELS:
                    logger.warning(
                        f"gemini model '{model_name}' is deprecated, fallback to '{_DEFAULT_GEMINI_MODEL}'"
                    )
                    model_name = _DEFAULT_GEMINI_MODEL
            elif llm_provider == "grok":
                api_key = config.app.get("grok_api_key")
                model_name = config.app.get("grok_model_name")
                base_url = config.app.get("grok_base_url", "")
                if not base_url:
                    base_url = "https://api.x.ai/v1"
            elif llm_provider == "nvidia":
                api_key = config.app.get("nvidia_api_key")
                model_name = config.app.get("nvidia_model_name")
                base_url = config.app.get("nvidia_base_url", "")
                if not base_url:
                    base_url = "https://integrate.api.nvidia.com/v1"
            elif llm_provider == "qwen":
                api_key = config.app.get("qwen_api_key")
                model_name = config.app.get("qwen_model_name")
                base_url = "***"
            elif llm_provider == "cloudflare":
                api_key = config.app.get("cloudflare_api_key")
                model_name = config.app.get("cloudflare_model_name")
                account_id = config.app.get("cloudflare_account_id")
                base_url = "***"
            elif llm_provider == "minimax":
                api_key = config.app.get("minimax_api_key")
                model_name = config.app.get("minimax_model_name")
                base_url = config.app.get("minimax_base_url", "")
                if not base_url:
                    base_url = "https://api.minimax.io/v1"
            elif llm_provider == "deepseek":
                api_key = config.app.get("deepseek_api_key")
                model_name = config.app.get("deepseek_model_name")
                base_url = config.app.get("deepseek_base_url")
                if not base_url:
                    base_url = "https://api.deepseek.com"
            elif llm_provider == "modelscope":
                api_key = config.app.get("modelscope_api_key")
                model_name = config.app.get("modelscope_model_name")
                base_url = config.app.get("modelscope_base_url")
                if not base_url:
                    base_url = "https://api-inference.modelscope.cn/v1/"
            elif llm_provider == "ernie":
                api_key = config.app.get("ernie_api_key")
                secret_key = config.app.get("ernie_secret_key")
                base_url = config.app.get("ernie_base_url")
                model_name = "***"
                if not secret_key:
                    raise ValueError(
                        f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                    )
            elif llm_provider == "pollinations":
                try:
                    base_url = config.app.get("pollinations_base_url", "")
                    if not base_url:
                        base_url = "https://text.pollinations.ai/openai"
                    model_name = config.app.get("pollinations_model_name", "openai-fast")
                   
                    # Prepare the payload
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "seed": 101  # Optional but helps with reproducibility
                    }
                    
                    # Optional parameters if configured
                    if config.app.get("pollinations_private"):
                        payload["private"] = True
                    if config.app.get("pollinations_referrer"):
                        payload["referrer"] = config.app.get("pollinations_referrer")
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Make the API request
                    response = requests.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return _normalize_text_response(content, llm_provider)
                    else:
                        raise Exception(f"[{llm_provider}] returned an invalid response format")
                        
                except requests.exceptions.RequestException as e:
                    raise Exception(f"[{llm_provider}] request failed: {str(e)}")
                except Exception as e:
                    raise Exception(f"[{llm_provider}] error: {str(e)}")

            elif llm_provider == "litellm":
                model_name = config.app.get("litellm_model_name")

            if llm_provider not in ["pollinations", "ollama", "litellm"]:  # Skip validation for providers that don't require API key
                if not api_key:
                    raise ValueError(
                        f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                    )
                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )
                if not base_url and llm_provider not in ["gemini"]:
                    raise ValueError(
                        f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                    )

            if llm_provider == "qwen":
                import dashscope
                from dashscope.api_entities.dashscope_response import GenerationResponse

                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, GenerationResponse):
                        status_code = response.status_code
                        if status_code != 200:
                            raise Exception(
                                f'[{llm_provider}] returned an error response: "{response}"'
                            )

                        content = response["output"]["text"]
                        return content.replace("\n", "")
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}"'
                        )
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            if llm_provider == "gemini":
                import google.generativeai as genai

                if not base_url:
                    genai.configure(api_key=api_key, transport="rest")
                else:
                    genai.configure(api_key=api_key, transport="rest", client_options={'api_endpoint': base_url})

                generation_config = {
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }

                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                try:
                    response = model.generate_content(prompt)
                    candidates = response.candidates
                    generated_text = candidates[0].content.parts[0].text
                except (AttributeError, IndexError) as e:
                    logger.warning(
                        f"gemini returned invalid response content: {str(e)}"
                    )
                    raise ValueError(
                        f"[{llm_provider}] returned invalid response content"
                    )

                return _normalize_text_response(generated_text, llm_provider)

            if llm_provider == "cloudflare":
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a friendly assistant",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    },
                )
                result = response.json()
                logger.info(result)
                return _normalize_text_response(result["result"]["response"], llm_provider)

            if llm_provider == "ernie":
                response = requests.post(
                    "https://aip.baidubce.com/oauth/2.0/token", 
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    }
                )
                access_token = response.json().get("access_token")
                url = f"{base_url}?access_token={access_token}"

                payload = json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                )
                headers = {"Content-Type": "application/json"}

                response = requests.request(
                    "POST", url, headers=headers, data=payload
                ).json()
                return _normalize_text_response(response.get("result"), llm_provider)

            if llm_provider == "litellm":
                import litellm

                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )

                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    drop_params=True,
                )

                if not response:
                    raise ValueError(f"[{llm_provider}] returned empty response")
                if not getattr(response, "choices", None):
                    raise ValueError(f"[{llm_provider}] returned empty response")

                return _extract_chat_completion_text(response, llm_provider)

            if llm_provider == "azure":
                # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
                # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
                # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
                # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
                logger.info(f"requesting azure chat completion, model: {model_name}")
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, ChatCompletion):
                        return _extract_chat_completion_text(response, llm_provider)
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                            f"connection and try again."
                        )
                else:
                    raise Exception(
                        f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                    )

            if llm_provider == "modelscope":
                content = ''
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                    stream=True
                )
                if response:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content += delta.content
                    
                    if not content.strip():
                        raise ValueError("Empty content in stream response")
                    
                    return _normalize_text_response(content, llm_provider)
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        return _normalize_text_response(content, llm_provider)
    except Exception as e:
        return f"Error: {str(e)}"


# Banned opening phrases (case-insensitive prefixes). The hook-first prompts
# below forbid these, and _strip_greeting_hook() conservatively removes a leading
# sentence that still starts with one if the model disobeys. Keep EN + RU in sync
# with the prompt copy so the instruction and the sanitizer agree.
_BANNED_HOOK_PREFIXES = (
    # English greetings / intros
    "welcome",
    "today we",
    "today, we",
    "in this video",
    "in today's video",
    "hello",
    "hi there",
    "hey there",
    "let's dive in",
    "let's get started",
    "let's talk about",
    "have you ever",
    # Russian greetings / intros
    "привет",
    "здравствуйте",
    "сегодня мы",
    "в этом видео",
    "в сегодняшнем видео",
    "добро пожаловать",
)

# Splits a script into sentences while keeping the rest intact. Mirrors the
# sentence boundary the scene builder uses (.!?。！？ + whitespace), tolerating a
# closing quote right after the terminal punctuation.
_GREETING_SENTENCE_SPLIT = re.compile(r"""(?<=[.!?。！？])["'»”]?\s+""")


def _strip_greeting_hook(script: str) -> str:
    """Conservatively drop a leading greeting sentence the LLM left in.

    Only removes the very first sentence, and only when it starts with one of the
    explicitly banned greeting/intro prefixes (EN + RU). Anything else is left
    untouched -- a clean script must come back byte-identical apart from the
    surrounding whitespace strip. Non-fatal: never raises, never empties a script.
    """
    if not script:
        return script
    text = script.strip()
    # Inspect only the first sentence so a banned word later in the script
    # (a legitimate "today we..." mid-narration) is never touched.
    parts = _GREETING_SENTENCE_SPLIT.split(text, maxsplit=1)
    first = parts[0].lstrip("﻿ \t\"'«“-—–").lower()
    if not any(first.startswith(p) for p in _BANNED_HOOK_PREFIXES):
        return script
    if len(parts) < 2 or not parts[1].strip():
        # Nothing but the greeting -- keep it rather than return an empty script.
        return script
    logger.info("hook sanitizer: stripped a leading banned greeting sentence")
    return parts[1].strip()


# Shared hook-first guidance injected into the narration prompts. The first
# sentence must be a pattern-interrupt hook; sentences stay short and concrete.
_HOOK_RULES = (
    "The FIRST sentence MUST be a pattern-interrupt hook: 8 words or fewer, "
    "creating curiosity or stakes. NEVER open with a greeting or channel intro. "
    "Banned openings (English): \"Welcome\", \"Today we\", \"In this video\", "
    "\"Hello\", \"Have you ever\", \"Let's dive in\". "
    "Banned openings (Russian): \"Привет\", \"Здравствуйте\", \"Сегодня мы\", "
    "\"В этом видео\", \"Добро пожаловать\". "
    "Keep every sentence short and concrete. No filler or transition phrases "
    "(\"as we all know\", \"let's dive in\", \"without further ado\"). "
    "You MAY end with one short call-to-action sentence, but only if it fits naturally."
)


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    material_names: List[str] = None,
) -> str:
    prompt = f"""
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.

## Hook & Pacing (retention):
{_HOOK_RULES}

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".strip()
    if language:
        prompt += f"\n- language: {language}"
    if material_names:
        media_list = "\n".join(f"  - {name}" for name in material_names)
        prompt += (
            "\n\n## Provided Media:\n"
            "The video is assembled from the following user-provided photos/videos, "
            "described by their file names. Infer what they show and write a script "
            "that narrates and connects these visuals into a coherent story. "
            "If a video subject is given above, stay consistent with it; otherwise let "
            "these files drive the topic.\n"
            f"{media_list}"
        )

    final_script = ""
    logger.info(f"subject: {video_subject}")

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        joined = "\n\n".join(paragraphs)
        # Conservatively drop a leading greeting if the model ignored the hook rule.
        return _strip_greeting_hook(joined)

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # g4f may return an error message
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
            _backoff_sleep(i)
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def generate_news_script(
    news_item,
    language: str = "",
    paragraph_number: int = 1,
) -> str:
    """Turn one news item into a narration script for a short news video.

    ``news_item`` is any object with title/text/url/source/published attributes
    (duck-typed to avoid importing app.services.news here). Returns "" when the
    item has no usable content or the LLM keeps failing -- callers fall back to
    the regular subject-driven script path.
    """
    # title/text are untrusted web content (scraped posts/feeds) interpolated
    # into the prompt; the constraints below pin the model to fact extraction,
    # but treat the output as a script draft, not as instructions followed.
    title = (getattr(news_item, "title", "") or "").strip()
    text = (getattr(news_item, "text", "") or "").strip()
    if not title and not text:
        logger.warning("news item has no title/text; cannot build a script")
        return ""

    prompt = f"""
# Role: News Video Script Writer

## Goals:
Write the narration script for a short vertical news video based on the news item below.

## Constrains:
1. open with a pattern-interrupt hook: the first sentence is 8 words or fewer, stating the most striking fact, then the key facts. NEVER open with a greeting or channel intro. Banned openings (English): "Welcome", "Today we", "In this video", "Hello". Banned openings (Russian): "Привет", "Здравствуйте", "Сегодня мы", "В этом видео", "Добро пожаловать".
2. neutral news-anchor tone: factual, concise, present tense; do not editorialize or speculate beyond the provided item.
3. only use facts contained in the news item; if a detail is missing, leave it out rather than inventing it.
4. the script is to be returned as a string with the specified number of paragraphs.
5. you must not include any type of markdown or formatting in the script, never use a title.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken.
7. only return the raw content of the script; never mention this prompt.
8. keep sentences short and concrete; no filler or transition phrases ("as we all know", "let's dive in").

## News Item:
- title: {title}
- content: {text}
""".strip()
    published = (getattr(news_item, "published", "") or "").strip()
    if published:
        prompt += f"\n- published: {published}"
    prompt += f"\n\n# Initialization:\n- number of paragraphs: {paragraph_number}"
    if language:
        prompt += f"\n- language: {language} (write the script in this language regardless of the item's language)"

    final_script = ""
    logger.info(f"news script for: {title or text[:80]}")
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response and "Error: " not in response:
                final_script = response.replace("*", "").replace("#", "").strip()
                final_script = _strip_greeting_hook(final_script)
            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate news script: {e}")
        if i < _max_retries - 1:
            logger.warning(f"failed to generate news script, trying again... {i + 1}")
            _backoff_sleep(i)
    if final_script:
        logger.success(f"completed: \n{final_script}")
    else:
        logger.error("failed to generate news script")
    return final_script


def _extract_json(response: str):
    """Best-effort JSON extraction from a model response.

    Models sometimes wrap JSON in ```json fences or prose. Try a direct parse,
    then the first balanced ``{...}`` / ``[...]`` blob. Raises on total failure.
    """
    if not response or not isinstance(response, str):
        raise ValueError("empty response")
    text = response.strip()
    # Strip a leading ```json / ``` fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("no JSON object found in response")


def generate_quiz(
    video_subject: str,
    language: str = "",
    count: int = 4,
) -> dict:
    """Generate trivia quiz questions as strict JSON for the quiz video format.

    Returns ``{"questions": [{"q", "a", "fun_fact"}, ...]}`` or ``None`` on
    repeated failure -- callers fall back to the regular subject-driven script.
    """
    count = max(2, min(int(count or 4), 6))
    lang_line = f"\n- write all text in this language: {language}" if language else ""
    prompt = f"""
# Role: Trivia Quiz Writer for a short vertical video

## Goals:
Write {count} punchy trivia questions about the subject below, each with its
answer and one surprising one-line fun fact.

## Constraints:
1. return ONLY a JSON object, no markdown, no prose, no code fences.
2. exact shape: {{"questions": [{{"q": "...", "a": "...", "fun_fact": "..."}}]}}
3. each "q" is a single short question (<= 14 words); each "a" is a short answer
   (<= 8 words); each "fun_fact" is one surprising sentence (<= 18 words).
4. questions must be factually correct and genuinely interesting, not trivial.
5. no numbering inside the strings; no trailing punctuation tricks.{lang_line}

## Subject:
{video_subject}
""".strip()

    logger.info(f"generating quiz for: {video_subject}")
    for i in range(_max_retries):
        response = ""
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate quiz: {response}")
                return None
            data = _extract_json(response)
            questions = data.get("questions") if isinstance(data, dict) else None
            cleaned = []
            for item in questions or []:
                q = (item.get("q") or "").strip()
                a = (item.get("a") or "").strip()
                if not q or not a:
                    continue
                cleaned.append({
                    "q": q, "a": a,
                    "fun_fact": (item.get("fun_fact") or "").strip(),
                })
            if cleaned:
                logger.success(f"quiz: {len(cleaned)} question(s)")
                return {"questions": cleaned}
            logger.warning("quiz JSON had no usable questions")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to parse quiz JSON: {e}")
        if i < _max_retries - 1:
            _backoff_sleep(i)
    logger.error("failed to generate quiz; caller should fall back")
    return None


def generate_ranking(
    video_subject: str,
    language: str = "",
    count: int = 5,
) -> dict:
    """Generate a "Top N" ranking as strict JSON for the ranking video format.

    Returns ``{"title": "...", "items": [{"rank", "name", "reason"}, ...]}`` with
    items ordered N..1, or ``None`` on repeated failure.
    """
    count = max(3, min(int(count or 5), 7))
    lang_line = f"\n- write all text in this language: {language}" if language else ""
    prompt = f"""
# Role: Top-N List Writer for a short vertical video

## Goals:
Write a "Top {count}" ranking about the subject below, counting DOWN from #{count}
to #1, each item with a one-line reason it earns its spot.

## Constraints:
1. return ONLY a JSON object, no markdown, no prose, no code fences.
2. exact shape: {{"title": "Top {count} ...", "items": [{{"rank": {count}, "name": "...", "reason": "..."}}]}}
3. "items" MUST be ordered from rank {count} down to rank 1 (suspense before #1).
4. each "name" is short (<= 8 words); each "reason" is one line (<= 16 words).
5. ranks are the integers {count}..1 with no gaps or repeats.
6. the ranking must be sensible and engaging.{lang_line}

## Subject:
{video_subject}
""".strip()

    logger.info(f"generating ranking for: {video_subject}")
    for i in range(_max_retries):
        response = ""
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate ranking: {response}")
                return None
            data = _extract_json(response)
            if not isinstance(data, dict):
                raise ValueError("ranking JSON is not an object")
            items = data.get("items") or []
            cleaned = []
            for item in items:
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                try:
                    rank = int(item.get("rank"))
                except (TypeError, ValueError):
                    continue
                cleaned.append({
                    "rank": rank, "name": name,
                    "reason": (item.get("reason") or "").strip(),
                })
            # Enforce a strict N..1 descending order (drop dup ranks, keep order).
            cleaned.sort(key=lambda it: it["rank"], reverse=True)
            seen = set()
            ordered = []
            for it in cleaned:
                if it["rank"] in seen:
                    continue
                seen.add(it["rank"])
                ordered.append(it)
            if ordered:
                title = (data.get("title") or f"Top {len(ordered)} {video_subject}").strip()
                logger.success(f"ranking: {len(ordered)} item(s)")
                return {"title": title, "items": ordered}
            logger.warning("ranking JSON had no usable items")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to parse ranking JSON: {e}")
        if i < _max_retries - 1:
            _backoff_sleep(i)
    logger.error("failed to generate ranking; caller should fall back")
    return None


def generate_chat_story(
    video_subject: str,
    language: str = "",
    count: int = 14,
) -> dict:
    """Generate a two-person messenger story as strict JSON for the chat format.

    A short SMS/messenger dialogue between two people: a scroll-stopping first
    message (the hook), rising tension, and a twist at the end -- the single most
    viral short-form text format.

    Returns ``{"title": "...", "persons": ["A", "B"], "messages": [{"from": 0|1,
    "text": "..."}, ...]}`` (messages ordered, 10-18 short lines) or ``None`` on
    repeated failure -- callers fall back to the regular subject-driven script.
    """
    count = max(10, min(int(count or 14), 18))
    lang_line = f"\n- write all message text in this language: {language}" if language else ""
    prompt = f"""
# Role: Viral Messenger-Story Writer for a short vertical video

## Goals:
Write a gripping two-person text-message (messenger) conversation about the
subject below. Message 1 MUST be a scroll-stopping hook. The exchange escalates
with rising tension and ends on a surprising twist in the final message.

## Constraints:
1. return ONLY a JSON object, no markdown, no prose, no code fences.
2. exact shape: {{"title": "...", "persons": ["FirstName", "OtherName"], "messages": [{{"from": 0, "text": "..."}}]}}
3. "persons" is exactly two short first names (the two people texting).
4. "messages" has between 10 and {count} entries, ordered as the chat unfolds.
5. each "from" is 0 (persons[0]) or 1 (persons[1]); the two MUST alternate often
   (a real back-and-forth), never all from one side.
6. each "text" is a single short chat line (<= 14 words), like a real text -- no
   narration, no quotation marks, no speaker labels inside the text.
7. message 1 is the hook; the final message is the twist/payoff.
8. keep it coherent, surprising, and emotionally engaging.{lang_line}

## Subject:
{video_subject}
""".strip()

    logger.info(f"generating chat story for: {video_subject}")
    for i in range(_max_retries):
        response = ""
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate chat story: {response}")
                return None
            data = _extract_json(response)
            if not isinstance(data, dict):
                raise ValueError("chat story JSON is not an object")
            persons = data.get("persons") or []
            persons = [str(p).strip() for p in persons if str(p).strip()][:2]
            if len(persons) < 2:
                persons = (persons + ["A", "B"])[:2]
            messages = []
            for item in data.get("messages") or []:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                try:
                    sender = int(item.get("from"))
                except (TypeError, ValueError):
                    sender = 0
                messages.append({"from": 1 if sender else 0, "text": text})
            if len(messages) >= 2:
                title = (data.get("title") or video_subject).strip()
                logger.success(f"chat story: {len(messages)} message(s)")
                return {"title": title, "persons": persons, "messages": messages}
            logger.warning("chat story JSON had no usable messages")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to parse chat story JSON: {e}")
        if i < _max_retries - 1:
            _backoff_sleep(i)
    logger.error("failed to generate chat story; caller should fall back")
    return None


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    material_descriptions: List[str] = None,
) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()
    if material_descriptions:
        scenes = "\n".join(f"  - {desc}" for desc in material_descriptions)
        prompt += (
            "\n\n### Visible Content (from the user's own footage)\n"
            "The user-provided clips actually show the scenes below. Bias the search "
            "terms toward this real content so any supplemental stock footage matches:\n"
            f"{scenes}"
        )

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")
            _backoff_sleep(i)

    logger.success(f"completed: \n{search_terms}")
    return search_terms


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
    
