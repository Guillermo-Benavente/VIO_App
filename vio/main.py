"""Run with: uvicorn vio.main:app --host 127.0.0.1 --port 8000"""

import uvicorn
from .api import create_app
from .core.config import settings

app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
