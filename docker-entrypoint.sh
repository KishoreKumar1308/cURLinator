#!/bin/bash
set -e

echo "🚀 Starting cURLinator API..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."

# Parse DATABASE_URL to extract host, port, and user
# DATABASE_URL format: postgresql://user:password@host:port/database
if [ -n "$DATABASE_URL" ]; then
    # Extract database connection details from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    DB_USER=$(echo $DATABASE_URL | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')

    echo "📊 Database connection details:"
    echo "   Host: $DB_HOST"
    echo "   Port: $DB_PORT"
    echo "   User: $DB_USER"

    # Wait for PostgreSQL to be ready using extracted values
    until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" 2>/dev/null; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
else
    # Fallback to individual environment variables (for local development)
    echo "⚠️  DATABASE_URL not set, using individual environment variables"
    until pg_isready -h "${DATABASE_HOST:-localhost}" -p "${DATABASE_PORT:-5432}" -U "${DATABASE_USER:-postgres}" 2>/dev/null; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
fi

echo "✅ PostgreSQL is ready!"

# Run database migrations
echo "🔄 Running database migrations..."
alembic upgrade head

echo "✅ Migrations complete!"

# Execute the main command (passed as arguments to this script)
echo "🎯 Starting application..."
exec "$@"

