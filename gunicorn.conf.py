import os
import multiprocessing

# Configuration Gunicorn
bind = "0.0.0.0:" + str(os.getenv("PORT", 8000))
workers = 1  # Limitez à 1 worker pour Railway
threads = 2
timeout = 30
worker_class = 'gthread'
worker_connections = 1000

# Limites de mémoire
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Limites de temps
graceful_timeout = 30
keep_alive = 5 