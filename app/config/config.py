import os
import shutil
import socket

import toml
from loguru import logger

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
env_file = f"{root_dir}/.env"


def load_dotenv():
    # Secrets live in a gitignored .env file so they never get committed.
    # We parse it manually (no extra dependency) and only set vars that are
    # not already present in the real environment, so OS-level env wins.
    if not os.path.isfile(env_file):
        return
    with open(env_file, mode="r", encoding="utf-8-sig") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/MoneyPrinterTurbo/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    try:
        _config_ = toml.load(config_file)
    except Exception as e:
        logger.warning(f"load config failed: {str(e)}, try to load as utf-8-sig")
        with open(config_file, mode="r", encoding="utf-8-sig") as fp:
            _cfg_content = fp.read()
            _config_ = toml.loads(_cfg_content)
    return _config_


def save_config():
    with open(config_file, "w", encoding="utf-8") as f:
        _cfg["app"] = app
        _cfg["azure"] = azure
        _cfg["siliconflow"] = siliconflow
        _cfg["ui"] = ui
        f.write(toml.dumps(_cfg))


load_dotenv()
_cfg = load_config()
app = _cfg.get("app", {})
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _cfg.get("azure", {})
siliconflow = _cfg.get("siliconflow", {})
# Product Telegram bot (app/bot). Must use its OWN bot token -- never reuse
# tokens belonging to other tooling on this machine.
telegram = _cfg.get("telegram", {})
ui = _cfg.get(
    "ui",
    {
        "hide_log": False,
    },
)


def apply_env_overrides():
    # Overlay secrets from the environment (populated by .env) on top of
    # config.toml. This lets the other PC just drop a .env in the project
    # root: config.toml carries non-secret settings, .env carries keys.
    # Scalar API keys: ENV_VAR_NAME == config_key.upper()
    scalar_app_keys = [
        "openai_api_key",
        "nvidia_api_key",
        "cloudflare_api_key",
        "cloudflare_account_id",
        "pollinations_api_key",
        "moonshot_api_key",
        "oneapi_api_key",
        "azure_api_key",
        "gemini_api_key",
        "grok_api_key",
        "qwen_api_key",
        "minimax_api_key",
        "deepseek_api_key",
        "modelscope_api_key",
        "ernie_api_key",
        "ernie_secret_key",
        "vision_api_key",
    ]
    for key in scalar_app_keys:
        env_val = os.getenv(key.upper())
        if env_val:
            app[key] = env_val

    # List-type keys accept a comma-separated value in .env
    list_app_keys = ["pexels_api_keys", "pixabay_api_keys"]
    for key in list_app_keys:
        env_val = os.getenv(key.upper())
        if env_val:
            app[key] = [item.strip() for item in env_val.split(",") if item.strip()]

    # Other sections
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.getenv("TELEGRAM_ALLOWED_CHATS"):
        # list[str] here vs ints/strings in TOML: the bot router compares
        # chat ids as strings, so both shapes converge.
        telegram["allowed_chats"] = [
            c.strip()
            for c in os.getenv("TELEGRAM_ALLOWED_CHATS").split(",")
            if c.strip()
        ]
    if os.getenv("AZURE_SPEECH_KEY"):
        azure["speech_key"] = os.getenv("AZURE_SPEECH_KEY")
    if os.getenv("AZURE_SPEECH_REGION"):
        azure["speech_region"] = os.getenv("AZURE_SPEECH_REGION")
    if os.getenv("SILICONFLOW_API_KEY"):
        siliconflow["api_key"] = os.getenv("SILICONFLOW_API_KEY")
    if os.getenv("UPLOAD_POST_API_KEY"):
        ui["upload_post_api_key"] = os.getenv("UPLOAD_POST_API_KEY")


apply_env_overrides()

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", "1.2.8")
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

imagemagick_path = app.get("imagemagick_path", "")
if imagemagick_path and os.path.isfile(imagemagick_path):
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
