#!/usr/bin/env python3
"""
Create Admin User Script

This script creates the first admin user for cURLinator.
Can be run locally or on Railway using: railway run python scripts/create_admin.py

Usage:
    # Interactive mode (prompts for email/password)
    python scripts/create_admin.py

    # Environment variable mode (non-interactive)
    ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD=SecurePass123 python scripts/create_admin.py

    # Railway deployment
    railway run python scripts/create_admin.py
"""

import os
import sys
import getpass
import time
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from curlinator.api.db.models import User
from curlinator.api.auth import get_password_hash


def get_database_url():
    """Get database URL from environment."""
    # Try DATABASE_URL_PUBLIC first (for Railway external access)
    database_url = os.getenv("DATABASE_URL_PUBLIC")

    # Fall back to DATABASE_URL
    if not database_url:
        database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("❌ Error: DATABASE_URL environment variable not set")
        print("\nPlease set DATABASE_URL to your PostgreSQL connection string:")
        print("export DATABASE_URL='postgresql://user:password@host:port/database'")
        print("\nFor Railway, use the public/external database URL:")
        print("export DATABASE_URL='postgresql://postgres:password@host.proxy.rlwy.net:port/railway'")
        sys.exit(1)

    # Fix Railway's postgres:// to postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    # Check if using internal Railway hostname (won't work from local machine)
    if "railway.internal" in database_url:
        print("⚠️  Warning: DATABASE_URL uses internal Railway hostname")
        print("   This only works from within Railway containers.")
        print("\nPlease use the public database URL instead:")
        print("   1. Go to Railway Dashboard → PostgreSQL service → Connect tab")
        print("   2. Copy the 'TCP Proxy Connection String' (host.proxy.rlwy.net)")
        print("   3. Run: DATABASE_URL='<public-url>' python scripts/create_admin.py")
        sys.exit(1)

    return database_url


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    return True, ""


def get_admin_credentials():
    """
    Get admin credentials from environment variables or user input.
    
    Returns:
        Tuple of (email, password)
    """
    # Try environment variables first (for non-interactive mode)
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")
    
    if email and password:
        print(f"📧 Using admin email from environment: {email}")
        is_valid, error = validate_password(password)
        if not is_valid:
            print(f"❌ Error: {error}")
            sys.exit(1)
        return email, password
    
    # Interactive mode
    print("\n" + "="*60)
    print("🔐 Create Admin User")
    print("="*60)
    print("\nThis script will create the first admin user for cURLinator.")
    print("\nPassword requirements:")
    print("  • At least 8 characters")
    print("  • At least one uppercase letter")
    print("  • At least one lowercase letter")
    print("  • At least one digit")
    print()
    
    # Get email
    while True:
        email = input("Admin email: ").strip()
        if email and "@" in email:
            break
        print("❌ Please enter a valid email address")
    
    # Get password
    while True:
        password = getpass.getpass("Admin password: ")
        is_valid, error = validate_password(password)
        
        if not is_valid:
            print(f"❌ {error}")
            continue
        
        # Confirm password
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("❌ Passwords do not match")
            continue
        
        break
    
    return email, password


def create_admin_user(email: str, password: str):
    """
    Create an admin user in the database.

    Args:
        email: Admin user email
        password: Admin user password (will be hashed)
    """
    database_url = get_database_url()

    print("\n🔄 Connecting to database...")

    # Retry logic for database connection
    max_retries = 5
    retry_delay = 1  # seconds

    db = None
    engine = None

    for attempt in range(1, max_retries + 1):
        try:
            # Create database engine and session
            engine = create_engine(database_url)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()

            # Test the connection
            db.execute(text("SELECT 1"))
            break  # Connection successful

        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️  Database connection failed (attempt {attempt}/{max_retries}). Retrying in {retry_delay}s...")
                print(f"   Error: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"\n❌ Failed to connect to database after {max_retries} attempts")
                print(f"   Error: {e}")
                sys.exit(1)

    if not db:
        print("\n❌ Failed to establish database connection")
        sys.exit(1)

    try:
        
        print("✅ Connected to database")
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        
        if existing_user:
            print(f"\n⚠️  User with email '{email}' already exists")
            
            # Check if already admin
            if existing_user.role == "admin":
                print(f"✅ User is already an admin")
                print(f"\nUser details:")
                print(f"  • ID: {existing_user.id}")
                print(f"  • Email: {existing_user.email}")
                print(f"  • Role: {existing_user.role}")
                print(f"  • Active: {existing_user.is_active}")
                db.close()
                return
            
            # Upgrade to admin
            response = input(f"\nUpgrade user '{email}' to admin role? (y/n): ").strip().lower()
            if response == 'y':
                existing_user.role = "admin"
                existing_user.is_active = True
                # Optionally update password
                update_password = input("Update password? (y/n): ").strip().lower()
                if update_password == 'y':
                    existing_user.hashed_password = get_password_hash(password)
                    print("🔐 Password updated")
                
                db.commit()
                print(f"\n✅ User '{email}' upgraded to admin!")
                print(f"\nUser details:")
                print(f"  • ID: {existing_user.id}")
                print(f"  • Email: {existing_user.email}")
                print(f"  • Role: {existing_user.role}")
                print(f"  • Active: {existing_user.is_active}")
            else:
                print("❌ Operation cancelled")
            
            db.close()
            return
        
        # Create new admin user
        print(f"\n🔄 Creating admin user '{email}'...")
        
        hashed_password = get_password_hash(password)
        
        admin_user = User(
            email=email,
            hashed_password=hashed_password,
            is_active=True,
            role="admin"
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print(f"\n✅ Admin user created successfully!")
        print(f"\nUser details:")
        print(f"  • ID: {admin_user.id}")
        print(f"  • Email: {admin_user.email}")
        print(f"  • Role: {admin_user.role}")
        print(f"  • Active: {admin_user.is_active}")
        
        print(f"\n🎉 You can now log in with:")
        print(f"   Email: {email}")
        print(f"   Password: <the password you entered>")
        
        db.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚀 cURLinator Admin User Creation Script")
    print("="*60)
    
    # Get credentials
    email, password = get_admin_credentials()
    
    # Create admin user
    create_admin_user(email, password)
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
