"""
Celery application configuration.
"""
from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "crm_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.celery.tasks.ai_tasks",
        "app.celery.tasks.lead_tasks",
        "app.celery.tasks.statistics_tasks",
    ],
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=240,  # 4 minutes soft limit
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        # Daily statistics at 8 AM UTC
        "daily-statistics": {
            "task": "app.celery.tasks.statistics_tasks.generate_daily_statistics",
            "schedule": crontab(hour=8, minute=0),
        },
        # Process stale leads every hour
        "process-stale-leads": {
            "task": "app.celery.tasks.lead_tasks.process_stale_leads",
            "schedule": crontab(minute=0),  # Every hour
        },
    },
)
