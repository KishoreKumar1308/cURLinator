"""OpenAPI/Swagger specification validation and detection"""

from enum import Enum


class OpenAPIVersion(str, Enum):
    """OpenAPI specification versions"""

    SWAGGER_2_0 = "2.0"
    OPENAPI_3_0 = "3.0"
    OPENAPI_3_1 = "3.1"
    UNKNOWN = "unknown"


class ValidationResult:
    """Result of OpenAPI validation"""

    def __init__(
        self,
        is_valid: bool,
        version: OpenAPIVersion = OpenAPIVersion.UNKNOWN,
        errors: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.version = version
        self.errors = errors or []

    def __repr__(self) -> str:
        return (
            f"ValidationResult(is_valid={self.is_valid}, "
            f"version={self.version}, errors={len(self.errors)})"
        )


def is_valid_openapi(spec: dict) -> bool:
    """
    Quick check if a dict is a valid OpenAPI specification.

    Args:
        spec: Dictionary to validate

    Returns:
        True if appears to be valid OpenAPI spec

    Examples:
        >>> is_valid_openapi({"openapi": "3.0.0", "info": {"title": "API"}})
        True
        >>> is_valid_openapi({"swagger": "2.0", "info": {"title": "API"}})
        True
        >>> is_valid_openapi({"random": "data"})
        False
    """
    if not isinstance(spec, dict):
        return False

    # Check for version identifiers
    has_openapi = "openapi" in spec
    has_swagger = "swagger" in spec

    if not (has_openapi or has_swagger):
        return False

    # Check for required 'info' field
    if "info" not in spec:
        return False

    # Validate info object has title
    info = spec.get("info", {})
    if not isinstance(info, dict) or "title" not in info:
        return False

    return True


def get_openapi_version(spec: dict) -> OpenAPIVersion:
    """
    Detect the OpenAPI/Swagger version from a specification.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        Detected OpenAPI version

    Examples:
        >>> get_openapi_version({"openapi": "3.0.0"})
        OpenAPIVersion.OPENAPI_3_0
        >>> get_openapi_version({"swagger": "2.0"})
        OpenAPIVersion.SWAGGER_2_0
    """
    # Check OpenAPI 3.x
    if "openapi" in spec:
        version_str = str(spec["openapi"])

        if version_str.startswith("3.1"):
            return OpenAPIVersion.OPENAPI_3_1
        elif version_str.startswith("3.0"):
            return OpenAPIVersion.OPENAPI_3_0
        elif version_str.startswith("3."):
            # Future 3.x versions, default to 3.1
            return OpenAPIVersion.OPENAPI_3_1

    # Check Swagger 2.0
    if "swagger" in spec:
        version_str = str(spec["swagger"])
        if version_str.startswith("2."):
            return OpenAPIVersion.SWAGGER_2_0

    return OpenAPIVersion.UNKNOWN


def validate_openapi_structure(spec: dict) -> ValidationResult:
    """
    Validate OpenAPI specification structure with detailed error reporting.

    Checks required fields based on detected version.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        ValidationResult with detailed validation information

    Examples:
        >>> result = validate_openapi_structure({"openapi": "3.0.0", "info": {"title": "API"}})
        >>> result.is_valid
        True
    """
    errors: list[str] = []

    # Basic type check
    if not isinstance(spec, dict):
        return ValidationResult(
            is_valid=False, errors=["Specification must be a dictionary/object"]
        )

    # Check for version identifier
    has_openapi = "openapi" in spec
    has_swagger = "swagger" in spec

    if not (has_openapi or has_swagger):
        errors.append("Missing 'openapi' or 'swagger' version field")
        return ValidationResult(is_valid=False, errors=errors)

    # Detect version
    version = get_openapi_version(spec)

    if version == OpenAPIVersion.UNKNOWN:
        errors.append("Unknown or unsupported OpenAPI version")
        return ValidationResult(is_valid=False, version=version, errors=errors)

    # Validate required fields based on version
    if version == OpenAPIVersion.SWAGGER_2_0:
        errors.extend(_validate_swagger_2_0(spec))
    else:  # OpenAPI 3.x
        errors.extend(_validate_openapi_3_x(spec))

    return ValidationResult(
        is_valid=len(errors) == 0,
        version=version,
        errors=errors,
    )


def _validate_swagger_2_0(spec: dict) -> list[str]:
    """Validate Swagger 2.0 required fields"""
    errors: list[str] = []

    # Required root fields
    required_fields = ["swagger", "info"]
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: '{field}'")

    # Validate info object
    if "info" in spec:
        info = spec["info"]
        if not isinstance(info, dict):
            errors.append("'info' must be an object")
        else:
            # Required info fields
            if "title" not in info:
                errors.append("Missing required field in info: 'title'")
            if "version" not in info:
                errors.append("Missing required field in info: 'version'")

    # Check paths or x-paths (some specs use extensions)
    if "paths" not in spec:
        errors.append("Missing 'paths' field (required in Swagger 2.0)")

    return errors


def _validate_openapi_3_x(spec: dict) -> list[str]:
    """Validate OpenAPI 3.x required fields"""
    errors: list[str] = []

    # Required root fields
    required_fields = ["openapi", "info"]
    for field in required_fields:
        if field not in spec:
            errors.append(f"Missing required field: '{field}'")

    # Validate info object
    if "info" in spec:
        info = spec["info"]
        if not isinstance(info, dict):
            errors.append("'info' must be an object")
        else:
            # Required info fields
            if "title" not in info:
                errors.append("Missing required field in info: 'title'")
            if "version" not in info:
                errors.append("Missing required field in info: 'version'")

    # Check for paths (recommended but not strictly required in 3.x)
    if "paths" not in spec and "components" not in spec and "webhooks" not in spec:
        errors.append("Specification should contain 'paths', 'components', or 'webhooks'")

    return errors


def extract_api_info(spec: dict) -> dict[str, str]:
    """
    Extract basic API information from OpenAPI spec.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        Dict with title, version, description, base_url

    Examples:
        >>> spec = {"openapi": "3.0.0", "info": {"title": "My API", "version": "1.0"}}
        >>> info = extract_api_info(spec)
        >>> info["title"]
        'My API'
    """
    result = {
        "title": "",
        "version": "",
        "description": "",
        "base_url": "",
    }

    if not isinstance(spec, dict):
        return result

    # Extract from info object
    info = spec.get("info", {})
    if isinstance(info, dict):
        result["title"] = info.get("title", "")
        result["version"] = info.get("version", "")
        result["description"] = info.get("description", "")

    # Try to extract base URL
    # Swagger 2.0
    if "host" in spec:
        scheme = spec.get("schemes", ["https"])[0] if "schemes" in spec else "https"
        host = spec["host"]
        base_path = spec.get("basePath", "")
        result["base_url"] = f"{scheme}://{host}{base_path}"

    # OpenAPI 3.x
    elif "servers" in spec:
        servers = spec["servers"]
        if isinstance(servers, list) and len(servers) > 0:
            result["base_url"] = servers[0].get("url", "")

    return result


def count_endpoints(spec: dict) -> int:
    """
    Count the number of endpoints in an OpenAPI spec.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        Number of endpoint+method combinations

    Examples:
        >>> count_endpoints({"paths": {"/users": {"get": {}, "post": {}}}})
        2
    """
    if not isinstance(spec, dict):
        return 0

    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return 0

    count = 0
    http_methods = ["get", "post", "put", "patch", "delete", "head", "options", "trace"]

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method in http_methods:
            if method in path_item:
                count += 1

    return count


def has_authentication(spec: dict) -> bool:
    """
    Check if OpenAPI spec defines authentication.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        True if authentication is defined
    """
    if not isinstance(spec, dict):
        return False

    # Swagger 2.0 - check securityDefinitions
    if "securityDefinitions" in spec:
        security_defs = spec["securityDefinitions"]
        if isinstance(security_defs, dict) and len(security_defs) > 0:
            return True

    # OpenAPI 3.x - check components.securitySchemes
    if "components" in spec:
        components = spec["components"]
        if isinstance(components, dict) and "securitySchemes" in components:
            security_schemes = components["securitySchemes"]
            if isinstance(security_schemes, dict) and len(security_schemes) > 0:
                return True

    # Check for global security requirement
    if "security" in spec:
        security = spec["security"]
        if isinstance(security, list) and len(security) > 0:
            return True

    return False


def get_spec_summary(spec: dict) -> dict[str, any]:
    """
    Get a summary of an OpenAPI specification.

    Args:
        spec: OpenAPI specification dictionary

    Returns:
        Dict with summary information
    """
    validation = validate_openapi_structure(spec)
    info = extract_api_info(spec)

    return {
        "is_valid": validation.is_valid,
        "version": validation.version.value,
        "title": info["title"],
        "api_version": info["version"],
        "description": info["description"],
        "base_url": info["base_url"],
        "endpoint_count": count_endpoints(spec),
        "has_authentication": has_authentication(spec),
        "validation_errors": validation.errors,
    }
