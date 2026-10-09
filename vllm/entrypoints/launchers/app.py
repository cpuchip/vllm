# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import re
import warnings
from argparse import Namespace

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from vllm.config import ModelConfig
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware
from vllm.entrypoints.serve.sagemaker.api_router import sagemaker_standards_bootstrap
from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins
from vllm.tasks import FALLBACK_SUPPORTED_TASKS, SupportedTask

from .api_server.routers import register_api_routers
from .utils.server_utils import lifespan


def build_app(
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
    model_config: ModelConfig | None = None,
) -> FastAPI:
    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "build_app and will be required in a future version. "
            "Defaulting to ('generate',).",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = FALLBACK_SUPPORTED_TASKS

    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.root_path = args.root_path

    # Compatibility endpoint for clients that probe the server root
    # before using the OpenAI-compatible API.
    @app.api_route(
        "/",
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    async def root():
        return {"status": "ok"}

    # Minimal Ollama compatibility endpoints used by `ollama launch codex`.
    @app.get("/api/status", include_in_schema=False)
    async def ollama_status():
        return {"status": "ok"}

    @app.get("/api/experimental/model-recommendations", include_in_schema=False)
    async def ollama_model_recommendations():
        return {"models": []}

    @app.post("/api/show", include_in_schema=False)
    async def ollama_show(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        req_model = str(body.get("model", "") or body.get("name", "")).strip()

        # Collect configured served model names and underlying model paths
        served_names: list[str] = []
        raw_served = getattr(args, "served_model_name", None)
        if raw_served:
            if isinstance(raw_served, list):
                served_names.extend(str(s) for s in raw_served)
            else:
                served_names.append(str(raw_served))
        if getattr(args, "model", None):
            served_names.append(str(args.model))
        if model_config and getattr(model_config, "model", None):
            served_names.append(str(model_config.model))

        # Deduplicate while preserving order
        unique_served: list[str] = list(dict.fromkeys(served_names))

        def _matches(candidate: str, target: str) -> bool:
            c = candidate.lower()
            t = target.lower()
            if c == t or c == os.path.basename(t):
                return True
            # Match colon tags or hyphens (e.g. qwen3.8:27b vs qwen3.8-27b)
            if c.replace(":", "-") == t.replace(":", "-"):
                return True
            # Match without :latest
            if c.removesuffix(":latest") == t or t.removesuffix(":latest") == c:
                return True
            # Prefix match before tag (e.g. qwen3.8 vs qwen3.8:27b)
            if c.split(":")[0] == t.split(":")[0]:
                return True
            return False

        matched = (
            any(_matches(req_model, name) for name in unique_served)
            if (unique_served and req_model)
            else True
        )

        if not matched:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"model '{req_model}' not found"
                },
            )

        # Context length from model_config, falling back to args or 262144
        context_len = (
            getattr(model_config, "max_model_len", None)
            or getattr(args, "max_model_len", None)
            or 262144
        )

        # Parameter size heuristic from model name if available
        param_match = re.search(
            r"(\d+(\.\d+)?[bB])",
            str(getattr(args, "model", "") or req_model),
        )
        param_size = param_match.group(1).upper() if param_match else "27B"

        return {
            "model_info": {
                "general.context_length": context_len,
            },
            "details": {
                "parameter_size": param_size,
            },
        }

    register_api_routers(args, app, supported_tasks, model_config)

    # Endpoint plugin hooks run after core routers so plugins can inspect them.
    # This runs even for the CPU only render server. A plugin eligible for
    # the `render` task still gets its routes registered. It receives
    # `engine_client=None` at Phase B (see `_init_endpoint_plugins_state`).
    attach_endpoint_plugins(app, supported_tasks)

    init_exception_handler(app)
    init_entrypoints_middleware(args, app, supported_tasks)
    app = sagemaker_standards_bootstrap(app)
    return app
