"""HTTP tool for making arbitrary HTTP requests."""

import aiohttp
from typing import Any, Dict

from .base import Tool, ToolResult
from .url_security import URLSecurityError, validate_public_http_url, validate_redirect_url


class HttpTool(Tool):
    """Make HTTP requests."""

    MAX_REDIRECTS = 5

    def __init__(self, work_dir=None):
        super().__init__(
            name="http",
            description="Make HTTP requests (GET, POST, PUT, DELETE). Basic SSRF protection included.",
        )
        self.work_dir = work_dir

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to send the request to",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default: GET)",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers as key-value pairs",
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        headers = params.get("headers", {})
        body = params.get("body", "")
        timeout = params.get("timeout", 30)

        if not url:
            return ToolResult(success=False, output="", error="Missing required parameter: url")

        try:
            validate_public_http_url(url)
        except URLSecurityError as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

        try:
            timeout = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                request_url = url
                request_method = method
                for redirect_count in range(self.MAX_REDIRECTS + 1):
                    kwargs: Dict[str, Any] = {"headers": headers, "allow_redirects": False}
                    if body and request_method in ("POST", "PUT", "PATCH"):
                        kwargs["data"] = body

                    async with session.request(request_method, request_url, **kwargs) as resp:
                        if resp.status in (301, 302, 303, 307, 308):
                            location = resp.headers.get("Location", "")
                            if redirect_count >= self.MAX_REDIRECTS:
                                return ToolResult(
                                    success=False,
                                    output="",
                                    error=f"Too many redirects while requesting: {url}",
                                )
                            try:
                                request_url = validate_redirect_url(str(resp.url), location)
                            except URLSecurityError as e:
                                return ToolResult(success=False, output="", error=str(e))
                            if resp.status == 303 and request_method not in ("GET", "HEAD"):
                                request_method = "GET"
                            continue

                        response_body = await resp.text()
                        status = resp.status

                        # Truncate large responses
                        max_chars = 50000
                        if len(response_body) > max_chars:
                            response_body = response_body[:max_chars] + "\n... [truncated]"

                        output_parts = [
                            f"HTTP {request_method} {request_url}",
                            f"Status: {status} {resp.reason}",
                            f"Headers: {dict(resp.headers)}",
                        ]
                        if response_body:
                            output_parts.append(f"\nBody:\n{response_body}")

                        return ToolResult(
                            success=status < 400,
                            output="\n".join(output_parts),
                            metadata={
                                "status": status,
                                "content_type": resp.headers.get("Content-Type", ""),
                                "final_url": str(resp.url),
                            },
                        )

                return ToolResult(
                    success=False,
                    output="",
                    error=f"Too many redirects while requesting: {url}",
                )

        except aiohttp.ClientError as e:
            return ToolResult(success=False, output="", error=f"HTTP client error: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"HTTP request failed: {e}")
