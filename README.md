# MLflow + FastAPI service

Сервис на FastAPI, который:
- при старте приложения загружает ML‑модель из MLflow
- имеет хэндлер `POST /predict` &mdash; принимает на вход признаки
- имеет хэндлер `POST /updateModel`, который принимает `run_id` и подменяет текущую модель на модель из этого run
- экспортирует Prometheus-метрики через `GET /metrics`
- накапливает батчи предсказаний и отправляет drift-отчёты в Evidently

## Переменные окружения

- **`MLFLOW_TRACKING_URI`**: URI вашего MLflow Tracking Server (например, `http://158.160.2.37:5000/`)
- **`DEFAULT_RUN_ID`**: модель загрузится из запуска с этим ID
- **`APP_PORT`**: внешний порт контейнера на ВМ, должен быть в диапазоне `8890-8899` для Prometheus scraping
- **`EVIDENTLY_URL`**: URL сервера Evidently
- **`EVIDENTLY_PROJECT_ID`**: ID проекта в Evidently для сохранения отчётов
- **`EVIDENTLY_BATCH_SIZE`**: размер батча событий для одного drift-отчёта
- **`EVIDENTLY_REPORT_INTERVAL_SECONDS`**: как часто фоновая задача проверяет готовность батча
- **`EVIDENTLY_ENABLED`**: включить / отключить отправку отчётов в Evidently

## Запуск

Через docker compose:

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=<your_run_id>
export APP_PORT=8890
docker compose up --build
```

Сервис будет доступен на `http://<ip>:8890`.

## Тесты

```bash
pytest -q
```

## Артефакты мониторинга

- импортируемый дашборд Grafana: `grafana/dashboard.json`
- набор алертов и выражений: `grafana/alerts.md`
- отчёт по доработкам: `report.md`
