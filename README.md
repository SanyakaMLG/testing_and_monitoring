# MLflow + FastAPI Service

В этом репозитории находится сервис на FastAPI, который загружает модель из MLflow, отдаёт предсказания по демографическим признакам, переключает активную модель по `run_id`, экспортирует Prometheus-метрики и отправляет drift отчёты в Evidently.

## Обработчики

- `POST /predict` принимает полный набор полей, оставляет только нужные активной модели и возвращает `prediction` и `probability`.
- `POST /updateModel` переключает текущую модель по `run_id`.
- `GET /health` показывает состояние сервиса, активный `run_id`, `model_type` и список обязательных фичей.
- `GET /metrics` экспортирует технические, дата- и model-метрики в формате Prometheus.
- Фоновая задача накапливает события инференса и отправляет Evidently drift-отчёты.

## Переменные окружения

- `MLFLOW_TRACKING_URI` — адрес MLflow Tracking Server. По умолчанию используется `http://158.160.2.37:5000/`.
- `DEFAULT_RUN_ID` — стартовая модель сервиса.
- `APP_PORT` — внешний порт контейнера. Для стенда используется диапазон `8890-8899`.
- `COMPOSE_PROJECT_NAME` — уникальное имя compose-проекта.
- `EVIDENTLY_URL` — адрес сервера Evidently.
- `EVIDENTLY_PROJECT_ID` — идентификатор проекта в Evidently.
- `EVIDENTLY_BATCH_SIZE` — размер батча для drift отчёта. По умолчанию `20`.
- `EVIDENTLY_REPORT_INTERVAL_SECONDS` — период фоновой проверки готовности батча. По умолчанию `30`.
- `EVIDENTLY_MAX_PENDING_EVENTS` — верхняя граница in-memory очереди событий для Evidently. По умолчанию `2000`.
- `EVIDENTLY_MAX_BATCHES_PER_INTERVAL` — сколько батчей можно обработать за один цикл фоновой задачи. По умолчанию `10`.
- `EVIDENTLY_ENABLED` — включает или выключает отправку отчётов в Evidently.

## Запуск сервиса

Обычно сервис запускается на ВМ так:

```bash
export DEFAULT_RUN_ID=985395b622c549b5b28b494d6e0248b2
export APP_PORT=8890
./scripts/run_service.sh
```

Если нужен ручной запуск:

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=985395b622c549b5b28b494d6e0248b2
export APP_PORT=8890
export COMPOSE_PROJECT_NAME=$(whoami | tr -c '[:alnum:]' '_' )_testing_and_monitoring
docker compose up --build -d
```

После старта сервис доступен на `http://<ip>:8890`.

## Примеры работы сервиса

Проверка состояния:

```bash
curl http://158.160.71.215:8890/health
```

Пример ответа:

```json
{
  "status": "ok",
  "run_id": "985395b622c549b5b28b494d6e0248b2",
  "model_type": "RandomForestClassifier",
  "features": [
    "race",
    "sex",
    "native.country",
    "occupation",
    "education",
    "capital.gain"
  ],
  "startup_error": null
}
```

Успешный прогноз:

```bash
curl -X POST http://158.160.71.215:8890/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "workclass": "Private",
    "fnlwgt": 190000,
    "education": "Bachelors",
    "education.num": 13,
    "marital.status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 45,
    "native.country": "United-States"
  }'
```

Пример ответа:

```json
{
  "prediction": 1,
  "probability": 0.6093315436980924
}
```

Пример ошибки валидации:

```bash
curl -X POST http://158.160.71.215:8890/predict \
  -H "Content-Type: application/json" \
  -d '{"age": -5}'
```

Переключение модели:

```bash
curl -X POST http://158.160.71.215:8890/updateModel \
  -H "Content-Type: application/json" \
  -d '{"run_id":"8990717746ed4cfda04aaabd43c8bad5"}'
```

Проверка экспорта метрик:

```bash
curl http://158.160.71.215:8890/metrics | head
```

## Нагрузочное тестирование

Короткий прогон:

```bash
python scripts/load_test.py \
  --base-url http://158.160.71.215:8890 \
  --duration 20 \
  --start-rps 10 \
  --end-rps 20 \
  --max-concurrency 80
```

Длинный прогон для заполнения Grafana:

```bash
python scripts/load_test.py \
  --base-url http://158.160.71.215:8890 \
  --duration 180 \
  --start-rps 10 \
  --end-rps 300 \
  --max-concurrency 400
```

## Мониторинг и алерты

Дашборд Grafana хранится в [grafana/dashboard.json](grafana/dashboard.json), а alert правила описаны в [grafana/alerts.md](grafana/alerts.md).

Уведомления отправляются через Telegram contact point `ML Service Telegram`.

## Drift Monitoring

Для Evidently реализованы bounded in-memory очередь событий и обработка нескольких батчей за один цикл фоновой задачи. Это позволяет не раздувать очередь бесконечно под высокой нагрузкой и быстрее разгребать backlog после всплесков трафика.

Быстрая проверка реальной отправки Evidently-отчётов:

```bash
scripts/test_evidently.sh \
  --base-url http://158.160.71.215:8890 \
  --batch-size 20 \
  --timeout 120
```

Скрипт сам:

- проверяет `/health`;
- снимает текущие Evidently метрики;
- отправляет достаточно валидных запросов в `/predict`;
- ждёт роста `ml_service_evidently_report_total{status="success"}`.

## Полезные файлы

- [report.md](report.md) — описание выполненных доработок.
- [grafana/dashboard.json](grafana/dashboard.json) — импортируемый дашборд Grafana.
- [grafana/alerts.md](grafana/alerts.md) — alert выражения и пороги.
- [scripts/run_service.sh](scripts/run_service.sh) — запуск сервиса на ВМ.
- [scripts/load_test.py](scripts/load_test.py) — генератор нагрузки.
- [scripts/test_evidently.sh](scripts/test_evidently.sh) — проверка фактической отправки Evidently отчётов.
