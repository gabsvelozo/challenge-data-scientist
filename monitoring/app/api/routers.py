"""Esse arquivo cria os endpoints da API"""
from fastapi import APIRouter
from .endpoints.performance import router as perf_router
from .endpoints.aderencia import router as ader_router
router = APIRouter()

# Rotas de cada endpoint
router.include_router(perf_router)
router.include_router(ader_router)
