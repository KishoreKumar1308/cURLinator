"""API specification data models"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HTTPMethod(str, Enum):
    """HTTP request methods"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthMethod(str, Enum):
    """Authentication methods"""

    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer"
    BASIC_AUTH = "basic"
    OAUTH2 = "oauth2"


class ParameterLocation(str, Enum):
    """Parameter location in request"""

    QUERY = "query"
    HEADER = "header"
    PATH = "path"
    BODY = "body"


class APIParameter(BaseModel):
    """API endpoint parameter"""

    name: str
    location: ParameterLocation
    type: str
    description: str | None = None
    required: bool = False
    default: Any | None = None
    example: Any | None = None


class APIEndpoint(BaseModel):
    """API endpoint specification"""

    path: str
    method: HTTPMethod
    summary: str | None = None
    description: str | None = None
    parameters: list[APIParameter] = Field(default_factory=list)
    request_body: dict[str, Any] | None = None
    responses: dict[str, Any] = Field(default_factory=dict)
    auth_required: bool = False
    tags: list[str] = Field(default_factory=list)


class APISpecification(BaseModel):
    """Complete API specification"""

    title: str
    version: str = "1.0.0"
    base_url: str
    description: str | None = None
    auth_methods: list[AuthMethod] = Field(default_factory=list)
    endpoints: list[APIEndpoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
