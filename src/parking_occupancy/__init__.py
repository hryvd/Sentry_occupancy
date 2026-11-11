"""
SENTRY Parking Occupancy Detection Service
===========================================
Module: parking_occupancy
Owner: Occupancy Team
Target: December 1–7, 2026 demonstration

3-camera perspective warp + YOLO vehicle detection pipeline.
Outputs per-space physical occupancy observations to FastAPI backend.

All outputs are OBSERVATIONS only — no reservation, release,
gate permission, or PostgreSQL access.
"""

__version__ = "0.1.0"

