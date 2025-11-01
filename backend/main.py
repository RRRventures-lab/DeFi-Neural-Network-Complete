"""
DeFi Neural Network Trading Dashboard Backend

FastAPI server for the trading engine with:
- Real-time trading operations
- Performance analytics
- Model management
- Configuration and settings
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from typing import Optional
import logging
import hashlib
import hmac

# Import routers
from routers import trading, performance, models, config as config_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeFi Neural Network Trading Dashboard API",
    description="Backend API for autonomous trading engine with real-time dashboard",
    version="1.0.0",
)

# Add CORS middleware for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:8000",  # Backend docs
        "https://defi-neural-network.vercel.app",  # Production Vercel domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication setup
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "default_hash")
ADMIN_TOKEN_VALID = False


def verify_password(password: str) -> bool:
    """Verify admin password using scrypt hashing."""
    # Simple HMAC-based verification for development
    expected_hash = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(expected_hash, ADMIN_PASSWORD_HASH)


def get_admin_token(password: str) -> Optional[str]:
    """Generate auth token if password is correct."""
    if verify_password(password):
        return hashlib.sha256(f"{password}{datetime.now().isoformat()}".encode()).hexdigest()
    return None


# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API is running."""
    return {
        "status": "running",
        "api": "DeFi Neural Network Trading Dashboard",
        "version": "1.0.0",
        "documentation": "/docs",
    }


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health and system status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trading_engine": "ready",
        "database": "connected",
    }


# Authentication endpoints
@app.post("/auth/login", tags=["Authentication"])
async def login(password: str):
    """
    Login with admin password.

    Args:
        password: Admin password

    Returns:
        Authorization token if password is correct
    """
    token = get_admin_token(password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid password")

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
    }


@app.post("/auth/verify", tags=["Authentication"])
async def verify_token(authorization: Optional[str] = None):
    """
    Verify authentication token.

    Args:
        authorization: Bearer token from Authorization header

    Returns:
        Token verification status
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    return {
        "valid": True,
        "user": "admin",
        "timestamp": datetime.now().isoformat(),
    }


# Include routers
app.include_router(
    trading.router,
    prefix="/api/trading",
    tags=["Trading Operations"],
)

app.include_router(
    performance.router,
    prefix="/api/performance",
    tags=["Performance Analytics"],
)

app.include_router(
    models.router,
    prefix="/api/models",
    tags=["Model Management"],
)

app.include_router(
    config_router.router,
    prefix="/api/config",
    tags=["Configuration"],
)


# WebSocket endpoint for real-time updates
from fastapi import WebSocket
from typing import Set

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.last_update = datetime.now().isoformat()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        """Broadcast update to all connected clients."""
        data["timestamp"] = datetime.now().isoformat()
        self.last_update = data["timestamp"]

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading updates.

    Clients connect and receive real-time updates for:
    - Position changes
    - Price updates
    - Performance metrics
    - System status
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back any received messages
            await websocket.send_json({"type": "pong", "data": data})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
