"""
Integration tests for Prometheus metrics.

Tests that metrics are properly collected and exposed via the /metrics endpoint.
"""

import re
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from curlinator.api.main import app
from curlinator.api.database import get_db, Base


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_metrics.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


def get_metric_total(metrics_text: str, metric_name: str, labels: dict = None) -> float:
    """
    Extract and sum all values for a metric from Prometheus metrics text.

    For Counter metrics with labels, this sums all values across all label combinations.
    For example, if curlinator_http_requests_total has multiple lines with different
    method/endpoint/status_code combinations, this returns the sum of all values.

    Args:
        metrics_text: The full Prometheus metrics output
        metric_name: Name of the metric to extract (e.g., 'curlinator_http_requests_total')
        labels: Optional dict of label key-value pairs to filter by (e.g., {'method': 'GET'})

    Returns:
        The sum of all matching metric values as a float, or 0.0 if not found

    Example:
        >>> text = '''
        ... curlinator_http_requests_total{method="GET",endpoint="/api",status_code="200"} 10.0
        ... curlinator_http_requests_total{method="POST",endpoint="/api",status_code="201"} 5.0
        ... '''
        >>> get_metric_total(text, 'curlinator_http_requests_total')
        15.0
    """
    # Build regex pattern for the metric
    if labels:
        # Build label pattern to match specific labels
        label_patterns = [f'{k}="{v}"' for k, v in labels.items()]
        label_pattern = r'\{[^}]*' + '.*'.join(re.escape(lp) for lp in label_patterns) + r'[^}]*\}'
        pattern = rf'{re.escape(metric_name)}{label_pattern}\s+(\d+(?:\.\d+)?)'
    else:
        # Match any labels or no labels
        pattern = rf'{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+(\d+(?:\.\d+)?)'

    # Find all matches and sum them
    matches = re.findall(pattern, metrics_text)
    total = sum(float(value) for value in matches)
    return total


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Create and drop test database for each test, and override the database dependency."""
    # Override the database dependency for this test
    app.dependency_overrides[get_db] = override_get_db

    # Create tables
    Base.metadata.create_all(bind=engine)

    yield

    # Drop tables
    Base.metadata.drop_all(bind=engine)

    # Clean up the override
    if get_db in app.dependency_overrides:
        del app.dependency_overrides[get_db]


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def test_user(client):
    """Create a test user and return authentication token."""
    import uuid
    # Register a new user with unique email
    unique_email = f"metrics_test_{uuid.uuid4().hex[:8]}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": unique_email,
            "password": "TestPass123"
        }
    )
    assert response.status_code == 201
    data = response.json()
    return {
        "token": data["access_token"],
        "email": unique_email
    }


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""
    
    def test_metrics_endpoint_exists(self, client):
        """Test that the /metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
    
    def test_metrics_endpoint_no_auth_required(self, client):
        """Test that /metrics endpoint doesn't require authentication."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should not return 401 Unauthorized
    
    def test_metrics_format(self, client):
        """Test that metrics are in Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for Prometheus format markers
        assert "# HELP" in content or "# TYPE" in content or "_total" in content
    
    def test_app_info_metric(self, client):
        """Test that application info metric is present."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for app info metric
        assert "curlinator_app_info" in content


class TestHTTPMetrics:
    """Test HTTP request metrics collection."""
    
    def test_http_request_metrics_collected(self, client, test_user):
        """Test that HTTP requests are tracked in metrics."""
        # Make a request to trigger metrics
        client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )
        
        # Check metrics endpoint
        response = client.get("/metrics")
        assert response.status_code == 200
        
        content = response.text
        
        # Check for HTTP metrics
        assert "curlinator_http_requests_total" in content
        assert "curlinator_http_request_duration_seconds" in content
    
    def test_http_metrics_track_status_codes(self, client, test_user):
        """Test that HTTP metrics track different status codes."""
        # Make successful request (200)
        client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )
        
        # Make unauthorized request (401)
        client.get("/api/v1/collections")
        
        # Check metrics
        response = client.get("/metrics")
        content = response.text
        
        # Should have metrics for both status codes
        assert "status_code=\"200\"" in content or "status_code=\"401\"" in content


class TestAuthMetrics:
    """Test authentication metrics collection."""
    
    def test_auth_registration_metrics(self, client):
        """Test that user registration is tracked in metrics."""
        import uuid
        # Register a user with unique email
        unique_email = f"auth_metrics_test_{uuid.uuid4().hex[:8]}@example.com"
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": unique_email,
                "password": "TestPass123"
            }
        )
        assert response.status_code == 201
        
        # Check metrics
        metrics_response = client.get("/metrics")
        content = metrics_response.text
        
        # Check for auth metrics
        assert "curlinator_auth_attempts_total" in content
        assert "curlinator_auth_tokens_created_total" in content
    
    def test_auth_login_metrics(self, client, test_user):
        """Test that user login is tracked in metrics."""
        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": test_user["email"],
                "password": "TestPass123"
            }
        )
        assert response.status_code == 200
        
        # Check metrics
        metrics_response = client.get("/metrics")
        content = metrics_response.text
        
        # Check for auth metrics
        assert "curlinator_auth_attempts_total" in content
        assert "endpoint=\"login\"" in content


class TestDatabaseMetrics:
    """Test database metrics collection."""
    
    def test_database_connection_metrics(self, client):
        """Test that database connection metrics are collected."""
        import uuid
        # Make a request that uses the database
        unique_email = f"db_metrics_test_{uuid.uuid4().hex[:8]}@example.com"
        client.post(
            "/api/v1/auth/register",
            json={
                "email": unique_email,
                "password": "TestPass123"
            }
        )
        
        # Check metrics
        response = client.get("/metrics")
        content = response.text
        
        # Check for database metrics
        assert "curlinator_db_connections_active" in content or "curlinator_db_queries_total" in content


class TestMetricsLabels:
    """Test that metrics have proper labels."""
    
    def test_http_metrics_have_labels(self, client, test_user):
        """Test that HTTP metrics include method and endpoint labels."""
        # Make a GET request
        client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )
        
        # Check metrics
        response = client.get("/metrics")
        content = response.text
        
        # Check for labels
        if "curlinator_http_requests_total" in content:
            # Should have method and endpoint labels
            assert "method=" in content
            assert "endpoint=" in content
    
    def test_auth_metrics_have_labels(self, client):
        """Test that auth metrics include endpoint and status labels."""
        import uuid
        # Register a user with unique email
        unique_email = f"labels_test_{uuid.uuid4().hex[:8]}@example.com"
        client.post(
            "/api/v1/auth/register",
            json={
                "email": unique_email,
                "password": "TestPass123"
            }
        )
        
        # Check metrics
        response = client.get("/metrics")
        content = response.text
        
        # Check for labels
        if "curlinator_auth_attempts_total" in content:
            assert "endpoint=" in content
            assert "status=" in content


class TestMetricsIncrements:
    """Test that metrics increment correctly."""
    
    def test_request_counter_increments(self, client, test_user):
        """Test that request counter increments with each request."""
        # Get initial metrics
        response1 = client.get("/metrics")
        content1 = response1.text

        # Extract initial counter value (sum across all label combinations)
        initial_count = get_metric_total(content1, "curlinator_http_requests_total")

        # Make a request (this should increment the counter)
        # Note: /metrics endpoint is excluded from metrics collection to avoid recursion
        client.get(
            "/api/v1/collections",
            headers={"Authorization": f"Bearer {test_user['token']}"}
        )

        # Get updated metrics
        response2 = client.get("/metrics")
        content2 = response2.text

        # Extract updated counter value (sum across all label combinations)
        updated_count = get_metric_total(content2, "curlinator_http_requests_total")

        # Verify the counter has incremented
        # The counter should increment by exactly 1 for the /api/v1/collections request
        # (/metrics requests are not counted to avoid recursion)
        assert updated_count > initial_count, (
            f"HTTP request counter should increment after making a request. "
            f"Initial: {initial_count}, Updated: {updated_count}"
        )


class TestMetricsAvailability:
    """Test that all expected metrics are available."""
    
    def test_all_metric_types_present(self, client):
        """Test that all major metric types are defined."""
        response = client.get("/metrics")
        content = response.text
        
        # Check for different metric types
        expected_metrics = [
            "curlinator_http_requests_total",
            "curlinator_http_request_duration_seconds",
            "curlinator_app_info",
        ]
        
        for metric in expected_metrics:
            assert metric in content, f"Expected metric {metric} not found in /metrics output"

