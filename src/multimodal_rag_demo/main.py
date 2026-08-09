from contextlib import asynccontextmanager

from fastapi import FastAPI

from multimodal_rag_demo.functions.controllers.debug_controller import router as debug_router
from multimodal_rag_demo.functions.controllers.query_controller import router as query_router
from multimodal_rag_demo.common.vertex_client import init_vertex


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_vertex()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(debug_router)
app.include_router(query_router)
