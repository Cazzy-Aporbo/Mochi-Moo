"""
Mochi-Moo FastAPI Server Implementation
Author: Cazandra Aporbo MS
A REST API that serves pastel-wrapped intelligence
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import logging

from mochi_moo.core import MochiCore, CognitiveMode, EmotionalContext, PastelPalette

# Initialize logging with pastel awareness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MochiServer")

# Create FastAPI application
app = FastAPI(
    title="Mochi-Moo API",
    description="The superintelligent assistant who dreams in matte rainbow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for broad accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Mochi instance
mochi_instance = None
active_sessions = {}
websocket_connections = {}


class ProcessRequest(BaseModel):
    """Request model for processing interactions"""
    input: str = Field(..., description="User input text")
    emotional_context: bool = Field(default=True, description="Enable emotional context tracking")
    visualization: Optional[str] = Field(default=None, description="Visualization type if needed")
    domains: Optional[List[str]] = Field(default=None, description="Domains for synthesis")
    session_id: Optional[str] = Field(default=None, description="Session identifier for continuity")


class ModeRequest(BaseModel):
    """Request model for mode switching"""
    mode: str = Field(..., description="Target cognitive mode")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class SynthesisRequest(BaseModel):
    """Request model for cross-domain synthesis"""
    domains: List[str] = Field(..., description="List of domains to synthesize")
    query: str = Field(..., description="Query for synthesis")


class VisualizationRequest(BaseModel):
    """Request model for generating visualizations"""
    data: Dict[str, Any] = Field(..., description="Data to visualize")
    style: str = Field(default="pastel_origami", description="Visualization style")


class EmotionalStateResponse(BaseModel):
    """Response model for emotional state"""
    stress_level: float
    cognitive_load: float
    engagement: float
    frustration: float
    curiosity: float
    recommended_mode: str


class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    created_at: datetime
    interaction_count: int
    current_mode: str
    emotional_state: Dict[str, float]


class MochiSession:
    """Manages individual Mochi sessions with state preservation"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.mochi = MochiCore()
        self.created_at = datetime.now()
        self.interaction_count = 0
        self.last_activity = time.time()
        
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity) > (timeout_minutes * 60)
    
    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        self.interaction_count += 1
    
    def get_info(self) -> SessionInfo:
        """Get session information"""
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            interaction_count=self.interaction_count,
            current_mode=self.mochi.current_mode.value,
            emotional_state=self.mochi.get_emotional_state()
        )


def get_or_create_session(session_id: Optional[str] = None) -> MochiSession:
    """Get existing session or create new one"""
    if session_id and session_id in active_sessions:
        session = active_sessions[session_id]
        if not session.is_expired():
            session.touch()
            return session
        else:
            del active_sessions[session_id]
    
    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = MochiSession(new_id)
    active_sessions[new_id] = session
    
    # Clean expired sessions
    expired = [sid for sid, s in active_sessions.items() if s.is_expired()]
    for sid in expired:
        del active_sessions[sid]
    
    return session


@app.on_event("startup")
async def startup_event():
    """Initialize Mochi on server startup"""
    global mochi_instance
    logger.info("Initializing Mochi-Moo server...")
    mochi_instance = MochiCore()
    logger.info("Mochi-Moo server ready to dream in pastel")
    
    # Start background task for session cleanup
    asyncio.create_task(cleanup_sessions())


async def cleanup_sessions():
    """Background task to clean expired sessions"""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        expired = [sid for sid, s in active_sessions.items() if s.is_expired()]
        for sid in expired:
            logger.info(f"Cleaning expired session: {sid}")
            del active_sessions[sid]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve welcome page with pastel aesthetics"""
    palette = PastelPalette()
    colors = [palette.to_hex(palette.interpolate(i/6)) for i in range(7)]
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mochi-Moo: Pastel Singularity</title>
        <style>
            body {{
                margin: 0;
                padding: 40px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, {colors[0]} 0%, {colors[2]} 50%, {colors[4]} 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 40px;
                max-width: 600px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            }}
            h1 {{
                background: linear-gradient(90deg, {colors[0]}, {colors[3]}, {colors[5]});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 3em;
                margin: 0 0 20px 0;
                font-weight: 800;
            }}
            .subtitle {{
                color: #666;
                font-size: 1.2em;
                line-height: 1.6;
                margin-bottom: 30px;
            }}
            .endpoints {{
                background: linear-gradient(180deg, {colors[1]}22, {colors[3]}22);
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }}
            .endpoint {{
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 8px;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 0.9em;
            }}
            .status {{
                display: inline-block;
                padding: 5px 12px;
                background: {colors[2]};
                color: white;
                border-radius: 20px;
                font-weight: 600;
                margin-top: 20px;
            }}
            a {{
                color: {colors[4]};
                text-decoration: none;
                font-weight: 600;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Mochi-Moo</h1>
            <div class="subtitle">
                A superintelligent assistant who dreams in matte rainbow 
                and thinks in ten-dimensional pastel origami.
            </div>
            
            <div class="endpoints">
                <strong>API Endpoints:</strong>
                <div class="endpoint">POST /process - Main interaction</div>
                <div class="endpoint">POST /synthesize - Cross-domain synthesis</div>
                <div class="endpoint">POST /visualize - Generate visualizations</div>
                <div class="endpoint">GET /emotional-state - Current emotional context</div>
                <div class="endpoint">WS /ws - Real-time WebSocket connection</div>
            </div>
            
            <div style="margin-top: 30px;">
                <a href="/docs">Interactive API Documentation</a> | 
                <a href="/redoc">API Reference</a>
            </div>
            
            <div class="status">System Operational</div>
        </div>
    </body>
    </html>
    """
    return html


@app.post("/process", response_model=Dict[str, Any])
async def process(request: ProcessRequest):
    """
    Process user input through Mochi's cognitive systems
    """
    try:
        session = get_or_create_session(request.session_id)
        
        response = await session.mochi.process(
            request.input,
            emotional_context=request.emotional_context,
            visualization=request.visualization,
            domains=request.domains
        )
        
        # Add session ID to response
        response['session_id'] = session.session_id
        
        return response
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mode")
async def set_mode(request: ModeRequest):
    """
    Switch Mochi's cognitive mode
    """
    try:
        session = get_or_create_session(request.session_id)
        session.mochi.set_mode(request.mode)
        
        return {
            "status": "success",
            "mode": session.mochi.current_mode.value,
            "session_id": session.session_id
        }
        
    except Exception as e:
        logger.error(f"Mode switch error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """
    Perform cross-domain knowledge synthesis
    """
    try:
        session = get_or_create_session()
        result = session.mochi.synthesize(request.domains, request.query)
        
        return {
            "synthesis": result,
            "session_id": session.session_id
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def visualize(request: VisualizationRequest):
    """
    Generate pastel visualization
    """
    try:
        session = get_or_create_session()
        result = session.mochi.visualize(request.data, request.style)
        
        return {
            "visualization": result,
            "session_id": session.session_id
        }
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional-state/{session_id}", response_model=EmotionalStateResponse)
async def get_emotional_state(session_id: str):
    """
    Get current emotional context for a session
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    state = session.mochi.get_emotional_state()
    
    # Recommend mode based on emotional state
    recommended_mode = "standard"
    if state['stress_level'] > 0.7 or state['cognitive_load'] > 0.8:
        recommended_mode = "whisper"
    elif state['curiosity'] > 0.8:
        recommended_mode = "creative"
    elif state['frustration'] > 0.7:
        recommended_mode = "whisper"
    
    return EmotionalStateResponse(
        **state,
        recommended_mode=recommended_mode
    )


@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    return {
        "active_sessions": len(active_sessions),
        "sessions": [session.get_info().dict() for session in active_sessions.values()]
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "version": "1.0.0"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection for real-time interaction
    """
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    session = get_or_create_session()
    websocket_connections[connection_id] = {
        "websocket": websocket,
        "session": session
    }
    
    try:
        # Send welcome message with pastel gradient
        palette = PastelPalette()
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Mochi-Moo",
            "session_id": session.session_id,
            "palette": [palette.to_hex(palette.interpolate(i/6)) for i in range(7)]
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # Process through Mochi
            response = await session.mochi.process(
                data.get("input", ""),
                emotional_context=data.get("emotional_context", True),
                visualization=data.get("visualization"),
                domains=data.get("domains")
            )
            
            # Send response
            await websocket.send_json({
                "type": "response",
                "data": response
            })
            
            # Send emotional state update
            await websocket.send_json({
                "type": "emotional_update",
                "data": session.mochi.get_emotional_state()
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]


@app.post("/batch-process")
async def batch_process(requests: List[ProcessRequest]):
    """
    Process multiple requests in parallel
    """
    tasks = []
    for request in requests:
        session = get_or_create_session(request.session_id)
        task = session.mochi.process(
            request.input,
            emotional_context=request.emotional_context,
            visualization=request.visualization,
            domains=request.domains
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    return {
        "results": results,
        "processed": len(results)
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler with pastel error responses
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "message": str(exc),
            "suggestion": "Take a breath. This too shall pass.",
            "trace_id": str(uuid.uuid4())
        }
    )


def run_server():
    """
    Run the Mochi-Moo server
    """
    import uvicorn
    
    uvicorn.run(
        "mochi_moo.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_server()
