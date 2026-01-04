"""
================================================================================
ALPHATRADE DEPLOYMENT & SETUP
================================================================================

Institutional-grade deployment and setup for the AlphaTrade trading system.

@infra: This script implements all deployment and infrastructure requirements:
  - Environment setup and validation
  - Docker container management
  - Kubernetes multi-region deployment (P3-D)
  - Database migration and initialization
  - Configuration management
  - Service health verification
  - WSL2 GPU environment setup

Commands:
    python main.py deploy setup              # Initial setup
    python main.py deploy docker up          # Start Docker services
    python main.py deploy docker down        # Stop Docker services
    python main.py deploy k8s apply          # Apply Kubernetes manifests
    python main.py deploy db migrate         # Run database migrations
    python main.py deploy env check          # Validate environment
    python main.py deploy gpu setup          # Setup GPU (WSL2)

Author: AlphaTrade System
Version: 1.3.0
================================================================================
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deploy")


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================


@dataclass
class DeployConfig:
    """Deployment configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    docker_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docker")
    k8s_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docker" / "kubernetes")

    # Environment
    env: str = "development"  # development, staging, production
    region: str = "US_EAST"  # US_EAST, US_WEST, EU_WEST, ASIA_PACIFIC

    # Docker
    compose_file: str = "docker-compose.yml"
    registry: str = ""

    # Kubernetes
    k8s_namespace: str = "alphatrade"
    k8s_context: str = ""

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "alphatrade"

    # WSL2/GPU
    wsl_distro: str = "Ubuntu-22.04"
    conda_env: str = "alphatrade"


# ============================================================================
# ENVIRONMENT VALIDATOR
# ============================================================================


class EnvironmentValidator:
    """
    Validate system environment for deployment.

    @infra: Ensures all prerequisites are met before deployment.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("EnvironmentValidator")
        self.checks: list[dict] = []

    def validate_all(self) -> bool:
        """Run all environment validations."""
        self.logger.info("Validating environment...")
        self.checks = []

        # Python version
        self._check_python_version()

        # Required packages
        self._check_required_packages()

        # Environment variables
        self._check_env_variables()

        # Docker
        self._check_docker()

        # Project structure
        self._check_project_structure()

        # Configuration files
        self._check_config_files()

        # Print results
        passed = all(c["passed"] for c in self.checks)

        print("\nEnvironment Validation Results:")
        print("=" * 60)
        for check in self.checks:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['name']}: {check['message']}")

        print("=" * 60)
        print(f"Overall: {'PASSED' if passed else 'FAILED'}")

        return passed

    def _check_python_version(self) -> None:
        """Check Python version >= 3.11."""
        import sys

        major, minor = sys.version_info[:2]
        passed = major >= 3 and minor >= 11

        self.checks.append({
            "name": "Python Version",
            "passed": passed,
            "message": f"{major}.{minor}" + ("" if passed else " (requires 3.11+)"),
        })

    def _check_required_packages(self) -> None:
        """Check required packages are installed."""
        required = [
            "numpy", "pandas", "pydantic", "xgboost", "lightgbm",
            "sklearn", "torch", "redis", "psycopg2",
        ]

        missing = []
        for pkg in required:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)

        passed = len(missing) == 0

        self.checks.append({
            "name": "Required Packages",
            "passed": passed,
            "message": "All installed" if passed else f"Missing: {', '.join(missing)}",
        })

    def _check_env_variables(self) -> None:
        """Check required environment variables."""
        required = ["ALPACA_API_KEY", "ALPACA_API_SECRET"]
        optional = ["DATABASE__HOST", "REDIS__HOST"]

        missing_required = [v for v in required if not os.getenv(v)]
        missing_optional = [v for v in optional if not os.getenv(v)]

        passed = len(missing_required) == 0

        if passed and missing_optional:
            message = f"OK (optional missing: {', '.join(missing_optional)})"
        elif passed:
            message = "All set"
        else:
            message = f"Missing: {', '.join(missing_required)}"

        self.checks.append({
            "name": "Environment Variables",
            "passed": passed,
            "message": message,
        })

    def _check_docker(self) -> None:
        """Check Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            passed = result.returncode == 0
            version = result.stdout.strip() if passed else "Not found"
        except Exception:
            passed = False
            version = "Not found"

        self.checks.append({
            "name": "Docker",
            "passed": passed,
            "message": version,
        })

    def _check_project_structure(self) -> None:
        """Check project directory structure."""
        required_dirs = [
            "quant_trading_system",
            "quant_trading_system/core",
            "quant_trading_system/models",
            "quant_trading_system/execution",
            "scripts",
            "data",
        ]

        missing = [d for d in required_dirs if not (self.config.project_root / d).exists()]

        self.checks.append({
            "name": "Project Structure",
            "passed": len(missing) == 0,
            "message": "Complete" if len(missing) == 0 else f"Missing: {', '.join(missing)}",
        })

    def _check_config_files(self) -> None:
        """Check configuration files exist."""
        config_files = [
            "pyproject.toml",
            "CLAUDE.md",
        ]

        optional_files = [
            ".env",
            "docker/docker-compose.yml",
        ]

        missing = [f for f in config_files if not (self.config.project_root / f).exists()]

        self.checks.append({
            "name": "Config Files",
            "passed": len(missing) == 0,
            "message": "Present" if len(missing) == 0 else f"Missing: {', '.join(missing)}",
        })


# ============================================================================
# DOCKER MANAGER
# ============================================================================


class DockerManager:
    """
    Manage Docker containers and services.

    @infra: Docker orchestration for the trading system.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("DockerManager")

    def up(self, services: list[str] | None = None, detach: bool = True) -> bool:
        """Start Docker services."""
        self.logger.info("Starting Docker services...")

        compose_path = self.config.docker_dir / self.config.compose_file

        if not compose_path.exists():
            self.logger.error(f"Docker Compose file not found: {compose_path}")
            return False

        cmd = ["docker-compose", "-f", str(compose_path), "up"]

        if detach:
            cmd.append("-d")

        if services:
            cmd.extend(services)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.config.docker_dir),
            )

            if result.returncode == 0:
                self.logger.info("Docker services started successfully")
                return True
            else:
                self.logger.error(f"Failed to start services: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Docker command failed: {e}")
            return False

    def down(self, remove_volumes: bool = False) -> bool:
        """Stop Docker services."""
        self.logger.info("Stopping Docker services...")

        compose_path = self.config.docker_dir / self.config.compose_file

        if not compose_path.exists():
            self.logger.error(f"Docker Compose file not found: {compose_path}")
            return False

        cmd = ["docker-compose", "-f", str(compose_path), "down"]

        if remove_volumes:
            cmd.append("-v")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.config.docker_dir),
            )

            if result.returncode == 0:
                self.logger.info("Docker services stopped")
                return True
            else:
                self.logger.error(f"Failed to stop services: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Docker command failed: {e}")
            return False

    def status(self) -> dict:
        """Get Docker service status."""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(self.config.docker_dir / self.config.compose_file), "ps", "--format", "json"],
                capture_output=True,
                text=True,
                cwd=str(self.config.docker_dir),
            )

            if result.returncode == 0 and result.stdout:
                # Parse JSON output
                services = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            services.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return {"services": services}

            return {"services": []}

        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}

    def logs(self, service: str | None = None, tail: int = 100) -> str:
        """Get Docker service logs."""
        cmd = ["docker-compose", "-f", str(self.config.docker_dir / self.config.compose_file),
               "logs", "--tail", str(tail)]

        if service:
            cmd.append(service)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.config.docker_dir),
            )
            return result.stdout

        except Exception as e:
            return f"Error: {e}"

    def build(self, services: list[str] | None = None, no_cache: bool = False) -> bool:
        """Build Docker images."""
        self.logger.info("Building Docker images...")

        cmd = ["docker-compose", "-f", str(self.config.docker_dir / self.config.compose_file), "build"]

        if no_cache:
            cmd.append("--no-cache")

        if services:
            cmd.extend(services)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.config.docker_dir),
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            return False


# ============================================================================
# KUBERNETES MANAGER
# ============================================================================


class KubernetesManager:
    """
    Manage Kubernetes deployments.

    @infra P3-D: Multi-region Kubernetes deployment support.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("KubernetesManager")

    def apply(self, manifest: str | None = None) -> bool:
        """Apply Kubernetes manifests."""
        self.logger.info("Applying Kubernetes manifests...")

        if not self.config.k8s_dir.exists():
            self.logger.error(f"Kubernetes directory not found: {self.config.k8s_dir}")
            return False

        if manifest:
            manifest_path = self.config.k8s_dir / manifest
            if not manifest_path.exists():
                self.logger.error(f"Manifest not found: {manifest_path}")
                return False
            manifests = [manifest_path]
        else:
            manifests = list(self.config.k8s_dir.glob("*.yaml"))

        success = True
        for m in manifests:
            cmd = ["kubectl", "apply", "-f", str(m)]

            if self.config.k8s_namespace:
                cmd.extend(["-n", self.config.k8s_namespace])

            if self.config.k8s_context:
                cmd.extend(["--context", self.config.k8s_context])

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"Applied: {m.name}")
                else:
                    self.logger.error(f"Failed to apply {m.name}: {result.stderr}")
                    success = False

            except Exception as e:
                self.logger.error(f"kubectl failed: {e}")
                success = False

        return success

    def delete(self, manifest: str | None = None) -> bool:
        """Delete Kubernetes resources."""
        self.logger.info("Deleting Kubernetes resources...")

        if manifest:
            manifest_path = self.config.k8s_dir / manifest
            if not manifest_path.exists():
                return False
            manifests = [manifest_path]
        else:
            manifests = list(self.config.k8s_dir.glob("*.yaml"))

        success = True
        for m in manifests:
            cmd = ["kubectl", "delete", "-f", str(m)]

            if self.config.k8s_namespace:
                cmd.extend(["-n", self.config.k8s_namespace])

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    success = False

            except Exception as e:
                self.logger.error(f"kubectl failed: {e}")
                success = False

        return success

    def get_pods(self) -> list[dict]:
        """Get pod status."""
        cmd = ["kubectl", "get", "pods", "-o", "json"]

        if self.config.k8s_namespace:
            cmd.extend(["-n", self.config.k8s_namespace])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("items", [])

        except Exception as e:
            self.logger.error(f"Failed to get pods: {e}")

        return []

    def create_namespace(self) -> bool:
        """Create Kubernetes namespace."""
        cmd = ["kubectl", "create", "namespace", self.config.k8s_namespace]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 or "already exists" in result.stderr

        except Exception:
            return False


# ============================================================================
# DATABASE MANAGER
# ============================================================================


class DatabaseManager:
    """
    Manage database migrations and setup.

    @infra: Database initialization and migration for TimescaleDB.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("DatabaseManager")

    def migrate(self) -> bool:
        """Run database migrations."""
        self.logger.info("Running database migrations...")

        try:
            from quant_trading_system.database.connection import get_connection
            from quant_trading_system.database.models import Base

            import asyncio

            async def run_migrations():
                engine = await get_connection()
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                return True

            return asyncio.run(run_migrations())

        except ImportError:
            self.logger.warning("Database module not available, skipping migrations")
            return True

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize database with required data."""
        self.logger.info("Initializing database...")

        # Create TimescaleDB hypertables if needed
        try:
            import asyncpg
            import asyncio

            async def init_db():
                conn = await asyncpg.connect(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    database=self.config.db_name,
                    user=os.getenv("DATABASE__USER", "postgres"),
                    password=os.getenv("DATABASE__PASSWORD", "postgres"),
                )

                # Enable TimescaleDB extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

                await conn.close()
                return True

            return asyncio.run(init_db())

        except Exception as e:
            self.logger.warning(f"Database initialization: {e}")
            return True  # Non-critical

    def check_connection(self) -> bool:
        """Test database connection."""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=os.getenv("DATABASE__USER", "postgres"),
                password=os.getenv("DATABASE__PASSWORD", "postgres"),
            )

            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()

            return True

        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False


# ============================================================================
# GPU SETUP (WSL2)
# ============================================================================


class GPUSetup:
    """
    Setup GPU environment for WSL2.

    @infra: Configure RAPIDS/cuDF for GPU-accelerated computation.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("GPUSetup")

    def check_wsl(self) -> bool:
        """Check if running in WSL2."""
        try:
            result = subprocess.run(
                ["wsl", "--status"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0

        except Exception:
            return False

    def check_gpu(self) -> dict:
        """Check GPU availability in WSL2."""
        info = {
            "available": False,
            "cuda_version": None,
            "gpu_name": None,
        }

        try:
            # Check nvidia-smi in WSL
            result = subprocess.run(
                ["wsl", "-d", self.config.wsl_distro, "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout:
                parts = result.stdout.strip().split(", ")
                info["available"] = True
                info["gpu_name"] = parts[0] if parts else None
                info["driver_version"] = parts[1] if len(parts) > 1 else None
                info["memory"] = parts[2] if len(parts) > 2 else None

        except Exception as e:
            self.logger.debug(f"GPU check failed: {e}")

        return info

    def setup_conda_env(self) -> bool:
        """Setup conda environment with RAPIDS in WSL2."""
        self.logger.info("Setting up conda environment in WSL2...")

        setup_script = f"""
source /root/miniconda3/bin/activate
conda activate {self.config.conda_env} || conda create -n {self.config.conda_env} python=3.11 -y
conda activate {self.config.conda_env}
pip install cudf-cu11 cuml-cu11 cugraph-cu11 cuspatial-cu11 cuproj-cu11 cuxfilter-cu11 cucim --extra-index-url=https://pypi.nvidia.com
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

        try:
            result = subprocess.run(
                ["wsl", "-d", self.config.wsl_distro, "bash", "-c", setup_script],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Conda setup failed: {e}")
            return False

    def sync_project(self) -> bool:
        """Sync project to WSL2."""
        self.logger.info("Syncing project to WSL2...")

        wsl_path = f"/root/AlphaTrade_System"

        try:
            # Create directory
            subprocess.run(
                ["wsl", "-d", self.config.wsl_distro, "mkdir", "-p", wsl_path],
                check=True,
            )

            # Sync files (using rsync-like approach)
            windows_path = str(self.config.project_root).replace("\\", "/")
            windows_path = f"/mnt/{windows_path[0].lower()}{windows_path[2:]}"

            sync_cmd = f"cp -r {windows_path}/* {wsl_path}/"

            subprocess.run(
                ["wsl", "-d", self.config.wsl_distro, "bash", "-c", sync_cmd],
                check=True,
            )

            return True

        except Exception as e:
            self.logger.error(f"Project sync failed: {e}")
            return False


# ============================================================================
# SETUP WIZARD
# ============================================================================


class SetupWizard:
    """
    Interactive setup wizard for initial deployment.

    @infra: Guides through complete system setup.
    """

    def __init__(self, config: DeployConfig):
        self.config = config
        self.logger = logging.getLogger("SetupWizard")

    def run(self) -> bool:
        """Run the setup wizard."""
        print("\n" + "=" * 60)
        print("ALPHATRADE SYSTEM SETUP WIZARD")
        print("=" * 60)

        steps = [
            ("Validate environment", self._step_validate),
            ("Create directories", self._step_directories),
            ("Setup configuration", self._step_config),
            ("Initialize database", self._step_database),
            ("Start services", self._step_services),
            ("Verify system", self._step_verify),
        ]

        for i, (name, step) in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] {name}...")
            try:
                success = step()
                if success:
                    print(f"  ✓ {name} complete")
                else:
                    print(f"  ✗ {name} failed")
                    if not self._prompt_continue():
                        return False
            except Exception as e:
                print(f"  ✗ Error: {e}")
                if not self._prompt_continue():
                    return False

        print("\n" + "=" * 60)
        print("SETUP COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. python main.py data download --symbols AAPL MSFT")
        print("  2. python main.py train --model xgboost")
        print("  3. python main.py trade --mode paper")

        return True

    def _prompt_continue(self) -> bool:
        """Prompt user to continue after failure."""
        response = input("Continue anyway? [y/N]: ").strip().lower()
        return response == "y"

    def _step_validate(self) -> bool:
        """Validate environment."""
        validator = EnvironmentValidator(self.config)
        return validator.validate_all()

    def _step_directories(self) -> bool:
        """Create required directories."""
        dirs = [
            "data/raw",
            "data/processed",
            "data/features",
            "models",
            "cache",
            "logs",
        ]

        for d in dirs:
            (self.config.project_root / d).mkdir(parents=True, exist_ok=True)

        return True

    def _step_config(self) -> bool:
        """Setup configuration files."""
        # Create .env if not exists
        env_file = self.config.project_root / ".env"

        if not env_file.exists():
            template = """# AlphaTrade Environment Configuration
# Copy this file to .env and fill in your values

# Alpaca API (Required)
ALPACA_API_KEY=your_api_key
ALPACA_API_SECRET=your_api_secret

# Database (Optional - for production)
DATABASE__HOST=localhost
DATABASE__PORT=5432
DATABASE__NAME=alphatrade
DATABASE__USER=postgres
DATABASE__PASSWORD=postgres

# Redis (Optional - for caching)
REDIS__HOST=localhost
REDIS__PORT=6379
"""
            env_file.write_text(template)
            print(f"  Created {env_file}")
            print("  Please edit .env with your API credentials")

        return True

    def _step_database(self) -> bool:
        """Initialize database."""
        db_manager = DatabaseManager(self.config)

        if db_manager.check_connection():
            return db_manager.migrate()
        else:
            print("  Database not available - skipping")
            return True

    def _step_services(self) -> bool:
        """Start Docker services if available."""
        docker = DockerManager(self.config)

        compose_file = self.config.docker_dir / self.config.compose_file
        if compose_file.exists():
            return docker.up()
        else:
            print("  Docker Compose not found - skipping")
            return True

    def _step_verify(self) -> bool:
        """Verify system is working."""
        from scripts.health import SystemHealthChecker
        import asyncio

        checker = SystemHealthChecker()

        async def quick_check():
            results = [
                await checker.check_data_feed(),
            ]
            return all(r.status.value in ["HEALTHY", "DEGRADED"] for r in results)

        return asyncio.run(quick_check())


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def cmd_setup(args: argparse.Namespace) -> int:
    """Run setup wizard."""
    config = DeployConfig()
    wizard = SetupWizard(config)

    success = wizard.run()
    return 0 if success else 1


def cmd_docker(args: argparse.Namespace) -> int:
    """Docker commands."""
    config = DeployConfig()
    docker = DockerManager(config)

    action = getattr(args, "docker_action", "status")

    if action == "up":
        services = getattr(args, "services", None)
        success = docker.up(services)
        return 0 if success else 1

    elif action == "down":
        remove_volumes = getattr(args, "volumes", False)
        success = docker.down(remove_volumes)
        return 0 if success else 1

    elif action == "status":
        status = docker.status()
        print(json.dumps(status, indent=2))
        return 0

    elif action == "logs":
        service = getattr(args, "service", None)
        tail = getattr(args, "tail", 100)
        logs = docker.logs(service, tail)
        print(logs)
        return 0

    elif action == "build":
        no_cache = getattr(args, "no_cache", False)
        success = docker.build(no_cache=no_cache)
        return 0 if success else 1

    return 1


def cmd_k8s(args: argparse.Namespace) -> int:
    """Kubernetes commands."""
    config = DeployConfig(
        k8s_namespace=getattr(args, "namespace", "alphatrade"),
        k8s_context=getattr(args, "context", ""),
    )
    k8s = KubernetesManager(config)

    action = getattr(args, "k8s_action", "status")

    if action == "apply":
        manifest = getattr(args, "manifest", None)
        success = k8s.apply(manifest)
        return 0 if success else 1

    elif action == "delete":
        manifest = getattr(args, "manifest", None)
        success = k8s.delete(manifest)
        return 0 if success else 1

    elif action == "status":
        pods = k8s.get_pods()
        for pod in pods:
            name = pod.get("metadata", {}).get("name", "unknown")
            phase = pod.get("status", {}).get("phase", "unknown")
            print(f"  {name}: {phase}")
        return 0

    return 1


def cmd_db(args: argparse.Namespace) -> int:
    """Database commands."""
    config = DeployConfig()
    db = DatabaseManager(config)

    action = getattr(args, "db_action", "check")

    if action == "migrate":
        success = db.migrate()
        return 0 if success else 1

    elif action == "init":
        success = db.initialize()
        return 0 if success else 1

    elif action == "check":
        success = db.check_connection()
        print(f"Database connection: {'OK' if success else 'FAILED'}")
        return 0 if success else 1

    return 1


def cmd_env(args: argparse.Namespace) -> int:
    """Environment commands."""
    config = DeployConfig()

    action = getattr(args, "env_action", "check")

    if action == "check":
        validator = EnvironmentValidator(config)
        success = validator.validate_all()
        return 0 if success else 1

    return 1


def cmd_gpu(args: argparse.Namespace) -> int:
    """GPU setup commands."""
    config = DeployConfig()
    gpu = GPUSetup(config)

    action = getattr(args, "gpu_action", "check")

    if action == "check":
        info = gpu.check_gpu()
        print("\nGPU Status:")
        print(f"  Available: {info['available']}")
        if info['available']:
            print(f"  Name: {info.get('gpu_name', 'unknown')}")
            print(f"  Driver: {info.get('driver_version', 'unknown')}")
            print(f"  Memory: {info.get('memory', 'unknown')}")
        return 0

    elif action == "setup":
        print("Setting up GPU environment in WSL2...")
        success = gpu.setup_conda_env()
        return 0 if success else 1

    elif action == "sync":
        success = gpu.sync_project()
        return 0 if success else 1

    return 1


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def run_deploy_command(args: argparse.Namespace) -> int:
    """
    Main entry point for deploy commands.

    @infra: This function routes to the appropriate deploy command handler.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    command = getattr(args, "deploy_command", "setup")

    commands = {
        "setup": cmd_setup,
        "docker": cmd_docker,
        "k8s": cmd_k8s,
        "db": cmd_db,
        "env": cmd_env,
        "gpu": cmd_gpu,
    }

    handler = commands.get(command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown deploy command: {command}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaTrade Deployment")
    subparsers = parser.add_subparsers(dest="deploy_command")

    # Setup command
    subparsers.add_parser("setup", help="Run setup wizard")

    # Docker commands
    docker_parser = subparsers.add_parser("docker", help="Docker operations")
    docker_parser.add_argument("docker_action", choices=["up", "down", "status", "logs", "build"])
    docker_parser.add_argument("--services", nargs="+")
    docker_parser.add_argument("--volumes", action="store_true")

    # K8s commands
    k8s_parser = subparsers.add_parser("k8s", help="Kubernetes operations")
    k8s_parser.add_argument("k8s_action", choices=["apply", "delete", "status"])
    k8s_parser.add_argument("--manifest", type=str)
    k8s_parser.add_argument("--namespace", type=str, default="alphatrade")

    # DB commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_parser.add_argument("db_action", choices=["migrate", "init", "check"])

    # Env commands
    env_parser = subparsers.add_parser("env", help="Environment validation")
    env_parser.add_argument("env_action", choices=["check"], default="check", nargs="?")

    # GPU commands
    gpu_parser = subparsers.add_parser("gpu", help="GPU setup")
    gpu_parser.add_argument("gpu_action", choices=["check", "setup", "sync"], default="check", nargs="?")

    args = parser.parse_args()
    sys.exit(run_deploy_command(args))
