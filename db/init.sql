-- Runs once, when the database is first created (an empty volume).
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    engine_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    rul DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL
);
