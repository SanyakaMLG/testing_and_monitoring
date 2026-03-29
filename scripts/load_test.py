#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx


DEFAULT_RUN_IDS = [
    'edfad2bc1f1e4681a4174ee5bb09bd35',
    '985395b622c549b5b28b494d6e0248b2',
    '8990717746ed4cfda04aaabd43c8bad5',
]
UNKNOWN_RUN_ID = 'deadbeefdeadbeefdeadbeefdeadbeef'

EDUCATION_LEVELS = [
    ('Preschool', 1),
    ('1st-4th', 2),
    ('5th-6th', 3),
    ('7th-8th', 4),
    ('9th', 5),
    ('10th', 6),
    ('11th', 7),
    ('12th', 8),
    ('HS-grad', 9),
    ('Some-college', 10),
    ('Assoc-voc', 11),
    ('Assoc-acdm', 12),
    ('Bachelors', 13),
    ('Masters', 14),
    ('Prof-school', 15),
    ('Doctorate', 16),
]
WORKCLASSES = [
    'Private',
    'Self-emp-not-inc',
    'Self-emp-inc',
    'Federal-gov',
    'Local-gov',
    'State-gov',
]
MARITAL_STATUSES = [
    'Never-married',
    'Married-civ-spouse',
    'Divorced',
    'Separated',
    'Widowed',
]
RELATIONSHIPS = [
    'Husband',
    'Wife',
    'Not-in-family',
    'Own-child',
    'Unmarried',
]
RACES = [
    'White',
    'Black',
    'Asian-Pac-Islander',
    'Amer-Indian-Eskimo',
    'Other',
]
SEXES = ['Male', 'Female']
COUNTRIES = [
    'United-States',
    'Mexico',
    'Canada',
    'India',
    'Philippines',
    'Germany',
    'Cuba',
    'Poland',
    'Jamaica',
]
HIGH_INCOME_OCCUPATIONS = [
    'Exec-managerial',
    'Prof-specialty',
    'Tech-support',
    'Sales',
]
MID_INCOME_OCCUPATIONS = [
    'Craft-repair',
    'Adm-clerical',
    'Protective-serv',
    'Transport-moving',
]
LOW_INCOME_OCCUPATIONS = [
    'Handlers-cleaners',
    'Machine-op-inspct',
    'Other-service',
    'Farming-fishing',
    'Priv-house-serv',
]


@dataclass
class RequestSpec:
    method: str
    path: str
    payload: dict[str, Any]
    scenario: str


@dataclass
class RuntimeState:
    run_ids: list[str]
    active_features: list[str] = field(default_factory=list)
    current_run_id: str | None = None
    next_run_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class Stats:
    started_at: float
    total_sent: int = 0
    scenario_counts: Counter[str] = field(default_factory=Counter)
    status_counts: Counter[str] = field(default_factory=Counter)
    endpoint_counts: Counter[str] = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, *, scenario: str, endpoint: str, status_code: int, latency_ms: float) -> None:
        async with self.lock:
            self.total_sent += 1
            self.scenario_counts[scenario] += 1
            self.status_counts[str(status_code)] += 1
            self.endpoint_counts[endpoint] += 1
            self.latencies_ms.append(latency_ms)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            latencies = self.latencies_ms.copy()
            snapshot = {
                'total_sent': self.total_sent,
                'scenario_counts': self.scenario_counts.copy(),
                'status_counts': self.status_counts.copy(),
                'endpoint_counts': self.endpoint_counts.copy(),
                'latencies_ms': latencies,
            }
        return snapshot


def choose_education(rng: random.Random, profile: str) -> tuple[str, int]:
    if profile == 'high':
        choices = EDUCATION_LEVELS[12:]
    elif profile == 'low':
        choices = EDUCATION_LEVELS[8:13]
    else:
        choices = EDUCATION_LEVELS[8:]
    return rng.choice(choices)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def build_valid_payload(rng: random.Random, profile: str) -> dict[str, Any]:
    education, education_num = choose_education(rng, profile)

    if profile == 'high':
        age = rng.randint(32, 68)
        occupation = rng.choice(HIGH_INCOME_OCCUPATIONS)
        hours = rng.randint(42, 70)
        capital_gain = rng.choice([0, 0, 0, rng.randint(1500, 25000)])
        capital_loss = rng.choice([0, 0, rng.randint(200, 2500)])
        marital_status = rng.choice(['Married-civ-spouse', 'Married-civ-spouse', 'Divorced'])
        relationship = rng.choice(['Husband', 'Wife', 'Not-in-family'])
        workclass = rng.choice(['Private', 'Self-emp-inc', 'Federal-gov'])
    elif profile == 'low':
        age = rng.randint(18, 45)
        occupation = rng.choice(LOW_INCOME_OCCUPATIONS)
        hours = rng.randint(15, 45)
        capital_gain = 0
        capital_loss = 0
        marital_status = rng.choice(['Never-married', 'Separated', 'Divorced'])
        relationship = rng.choice(['Own-child', 'Unmarried', 'Not-in-family'])
        workclass = rng.choice(['Private', 'Local-gov', 'State-gov'])
    else:
        age = rng.randint(24, 60)
        occupation = rng.choice(MID_INCOME_OCCUPATIONS + HIGH_INCOME_OCCUPATIONS + LOW_INCOME_OCCUPATIONS)
        hours = rng.randint(25, 60)
        capital_gain = rng.choice([0, 0, rng.randint(500, 7000)])
        capital_loss = rng.choice([0, 0, rng.randint(100, 2000)])
        marital_status = rng.choice(MARITAL_STATUSES)
        relationship = rng.choice(RELATIONSHIPS)
        workclass = rng.choice(WORKCLASSES)

    fnlwgt = clamp(int(rng.gauss(190_000, 75_000)), 20_000, 600_000)

    return {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': rng.choice(RACES),
        'sex': rng.choice(SEXES),
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours,
        'native.country': rng.choice(COUNTRIES),
    }


async def refresh_health(client: httpx.AsyncClient, state: RuntimeState) -> None:
    try:
        response = await client.get('/health')
        response.raise_for_status()
    except httpx.HTTPError:
        return

    data = response.json()
    async with state.lock:
        state.active_features = list(data.get('features') or [])
        state.current_run_id = data.get('run_id')


async def build_request_for_scenario(
    *,
    scenario: str,
    client: httpx.AsyncClient,
    state: RuntimeState,
    rng: random.Random,
) -> RequestSpec:
    if scenario == 'predict_valid_low':
        return RequestSpec('POST', '/predict', build_valid_payload(rng, 'low'), scenario)

    if scenario == 'predict_valid_mid':
        return RequestSpec('POST', '/predict', build_valid_payload(rng, 'mid'), scenario)

    if scenario == 'predict_valid_high':
        return RequestSpec('POST', '/predict', build_valid_payload(rng, 'high'), scenario)

    if scenario == 'predict_missing_feature':
        payload = build_valid_payload(rng, rng.choice(['low', 'mid', 'high']))
        async with state.lock:
            active_features = state.active_features.copy()
        if not active_features:
            await refresh_health(client, state)
            async with state.lock:
                active_features = state.active_features.copy()
        removable = [feature for feature in active_features if feature in payload]
        if not removable:
            removable = ['age', 'education.num', 'hours.per.week']
        payload.pop(rng.choice(removable), None)
        return RequestSpec('POST', '/predict', payload, scenario)

    if scenario == 'predict_negative_age':
        payload = build_valid_payload(rng, 'low')
        payload['age'] = -rng.randint(1, 10)
        return RequestSpec('POST', '/predict', payload, scenario)

    if scenario == 'predict_hours_too_high':
        payload = build_valid_payload(rng, 'mid')
        payload['hours.per.week'] = rng.randint(169, 220)
        return RequestSpec('POST', '/predict', payload, scenario)

    if scenario == 'predict_extra_field':
        payload = build_valid_payload(rng, 'mid')
        payload['unexpected_feature'] = 'boom'
        return RequestSpec('POST', '/predict', payload, scenario)

    if scenario == 'predict_wrong_type':
        payload = build_valid_payload(rng, 'high')
        payload['education.num'] = 'not-an-int'
        return RequestSpec('POST', '/predict', payload, scenario)

    if scenario == 'predict_empty_body':
        return RequestSpec('POST', '/predict', {}, scenario)

    if scenario == 'update_invalid_run_id':
        return RequestSpec('POST', '/updateModel', {'run_id': 'not-a-valid-run-id'}, scenario)

    if scenario == 'update_unknown_run_id':
        return RequestSpec('POST', '/updateModel', {'run_id': UNKNOWN_RUN_ID}, scenario)

    if scenario == 'update_existing_run':
        async with state.lock:
            run_ids = state.run_ids.copy()
            current_run_id = state.current_run_id
            index = state.next_run_index

        if not run_ids:
            run_ids = DEFAULT_RUN_IDS.copy()

        for _ in range(len(run_ids)):
            candidate = run_ids[index % len(run_ids)]
            index += 1
            if candidate != current_run_id:
                async with state.lock:
                    state.next_run_index = index
                return RequestSpec('POST', '/updateModel', {'run_id': candidate}, scenario)

        async with state.lock:
            state.next_run_index = index
        return RequestSpec('POST', '/updateModel', {'run_id': run_ids[0]}, scenario)

    raise ValueError(f'Unknown scenario: {scenario}')


async def send_request(
    *,
    client: httpx.AsyncClient,
    state: RuntimeState,
    stats: Stats,
    semaphore: asyncio.Semaphore,
    scenario: str,
    rng_seed: int,
) -> None:
    rng = random.Random(rng_seed)
    spec = await build_request_for_scenario(
        scenario=scenario,
        client=client,
        state=state,
        rng=rng,
    )

    start = time.perf_counter()
    status_code = 599
    async with semaphore:
        try:
            response = await client.request(spec.method, spec.path, json=spec.payload)
            status_code = response.status_code
            if spec.path == '/updateModel' and response.status_code == 200:
                await refresh_health(client, state)
        except httpx.HTTPError:
            status_code = 599

    latency_ms = (time.perf_counter() - start) * 1000
    await stats.record(
        scenario=spec.scenario,
        endpoint=spec.path,
        status_code=status_code,
        latency_ms=latency_ms,
    )


def pick_scenario(rng: random.Random) -> str:
    scenarios = [
        ('predict_valid_low', 22),
        ('predict_valid_mid', 24),
        ('predict_valid_high', 24),
        ('predict_missing_feature', 6),
        ('predict_negative_age', 4),
        ('predict_hours_too_high', 4),
        ('predict_extra_field', 4),
        ('predict_wrong_type', 4),
        ('predict_empty_body', 3),
        ('update_invalid_run_id', 2),
        ('update_unknown_run_id', 2),
        ('update_existing_run', 5),
    ]
    population = [name for name, _ in scenarios]
    weights = [weight for _, weight in scenarios]
    return rng.choices(population, weights=weights, k=1)[0]


async def warm_up_scenarios(
    *,
    client: httpx.AsyncClient,
    state: RuntimeState,
    stats: Stats,
    semaphore: asyncio.Semaphore,
    seed: int,
) -> None:
    primer_scenarios = [
        'predict_valid_low',
        'predict_valid_mid',
        'predict_valid_high',
        'predict_missing_feature',
        'predict_negative_age',
        'predict_hours_too_high',
        'predict_extra_field',
        'predict_wrong_type',
        'predict_empty_body',
        'update_invalid_run_id',
        'update_unknown_run_id',
        'update_existing_run',
    ]
    for offset, scenario in enumerate(primer_scenarios):
        await send_request(
            client=client,
            state=state,
            stats=stats,
            semaphore=semaphore,
            scenario=scenario,
            rng_seed=seed + offset,
        )


async def report_progress(stats: Stats, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(5)
        snapshot = await stats.snapshot()
        total = snapshot['total_sent']
        elapsed = max(time.perf_counter() - stats.started_at, 0.001)
        status_counts = ', '.join(f'{code}={count}' for code, count in sorted(snapshot['status_counts'].items()))
        print(f'[progress] sent={total} avg_rps={total / elapsed:.1f} statuses=[{status_counts}]', flush=True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


async def run_load(args: argparse.Namespace) -> None:
    state = RuntimeState(run_ids=args.run_ids.copy())
    stats = Stats(started_at=time.perf_counter())
    semaphore = asyncio.Semaphore(args.max_concurrency)

    limits = httpx.Limits(
        max_keepalive_connections=args.max_concurrency,
        max_connections=args.max_concurrency * 2,
    )
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip('/'), timeout=timeout, limits=limits) as client:
        await refresh_health(client, state)
        await warm_up_scenarios(
            client=client,
            state=state,
            stats=stats,
            semaphore=semaphore,
            seed=args.seed,
        )

        stop_event = asyncio.Event()
        reporter = asyncio.create_task(report_progress(stats, stop_event))
        tasks: set[asyncio.Task[None]] = set()
        rng = random.Random(args.seed)
        request_index = 0
        started_at = time.perf_counter()
        scheduled_at = started_at

        try:
            while True:
                now = time.perf_counter()
                elapsed = now - started_at
                if elapsed >= args.duration:
                    break

                progress = elapsed / args.duration if args.duration > 0 else 1.0
                target_rps = args.start_rps + (args.end_rps - args.start_rps) * progress
                interval = 1.0 / max(target_rps, 1.0)
                scenario = pick_scenario(rng)

                task = asyncio.create_task(
                    send_request(
                        client=client,
                        state=state,
                        stats=stats,
                        semaphore=semaphore,
                        scenario=scenario,
                        rng_seed=args.seed + request_index,
                    ),
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                request_index += 1

                scheduled_at += interval
                await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            stop_event.set()
            await reporter

    snapshot = await stats.snapshot()
    latencies = sorted(snapshot['latencies_ms'])
    total = snapshot['total_sent']
    elapsed = max(time.perf_counter() - stats.started_at, 0.001)

    print('\n=== Load Test Summary ===', flush=True)
    print(f'base_url: {args.base_url}', flush=True)
    print(f'duration: {args.duration}s', flush=True)
    print(f'sent: {total}', flush=True)
    print(f'avg_rps: {total / elapsed:.1f}', flush=True)
    if latencies:
        print(f'latency avg: {statistics.mean(latencies):.1f} ms', flush=True)
        print(f'latency p50: {percentile(latencies, 0.50):.1f} ms', flush=True)
        print(f'latency p90: {percentile(latencies, 0.90):.1f} ms', flush=True)
        print(f'latency p95: {percentile(latencies, 0.95):.1f} ms', flush=True)
        print(f'latency p99: {percentile(latencies, 0.99):.1f} ms', flush=True)

    print('\nstatus counts:', flush=True)
    for status_code, count in sorted(snapshot['status_counts'].items()):
        print(f'  {status_code}: {count}', flush=True)

    print('\nscenario counts:', flush=True)
    for scenario, count in snapshot['scenario_counts'].most_common():
        print(f'  {scenario}: {count}', flush=True)

    print('\nendpoint counts:', flush=True)
    for endpoint, count in snapshot['endpoint_counts'].most_common():
        print(f'  {endpoint}: {count}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Variable-RPS load generator for the ML FastAPI service.',
    )
    parser.add_argument(
        '--base-url',
        default='http://158.160.71.215:8890',
        help='Base URL of the deployed service.',
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=180,
        help='Load duration in seconds.',
    )
    parser.add_argument(
        '--start-rps',
        type=float,
        default=10.0,
        help='Starting requests per second.',
    )
    parser.add_argument(
        '--end-rps',
        type=float,
        default=300.0,
        help='Final requests per second.',
    )
    parser.add_argument(
        '--max-concurrency',
        type=int,
        default=400,
        help='Maximum number of in-flight requests.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible request generation.',
    )
    parser.add_argument(
        '--run-id',
        action='append',
        dest='run_ids',
        default=None,
        help='Known valid run_id for successful /updateModel traffic. Can be passed multiple times.',
    )
    args = parser.parse_args()
    if not args.run_ids:
        args.run_ids = DEFAULT_RUN_IDS.copy()
    return args


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_load(args))
    except KeyboardInterrupt:
        print('\nInterrupted by user', flush=True)


if __name__ == '__main__':
    main()
