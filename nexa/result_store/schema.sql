-- Flex result store. Large KG query results are kept here, out of the LLM
-- context: the agent only sees a result_id and a preview, and reads further
-- through the result tools (read_result, summarize_result, ...).

CREATE TABLE IF NOT EXISTS nexa_v2_results (
    result_id          UUID PRIMARY KEY,
    thread_id          TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cypher_query       TEXT,
    source_tool        TEXT NOT NULL,
    parent_result_id   UUID NULL,
    columns_schema     JSONB NOT NULL,
    row_count          INTEGER NOT NULL,
    data               JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nexa_v2_results_thread  ON nexa_v2_results(thread_id);
CREATE INDEX IF NOT EXISTS idx_nexa_v2_results_created ON nexa_v2_results(created_at);

-- Completeness verdict of the execute_cypher completeness guard
-- (NEXA_COMPLETENESS_GUARD):
--   NULL                      not assessed (guard off or no trailing integer LIMIT),
--                             which is not a claim of completeness
--   {"complete": true}        assessed and complete
--   {"complete": false, ...}  the query returned exactly its own LIMIT and more rows exist
-- Stored with the result so it survives pagination and thread reloads.
ALTER TABLE nexa_v2_results ADD COLUMN IF NOT EXISTS completeness JSONB NULL;

-- Agent mode ("direct" or "flex") each thread was created in, so the frontend
-- renders the thread in that mode. Written once per thread (ON CONFLICT DO NOTHING).
CREATE TABLE IF NOT EXISTS nexa_threads (
    thread_id   TEXT PRIMARY KEY,
    agent_mode  TEXT NOT NULL CHECK (agent_mode IN ('direct', 'flex')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Map rows stored with the alias "agentic" to "flex" (idempotent). The inline
-- CHECK constraint (Postgres's default name) is dropped, the rows rewritten,
-- and the constraint re-added.
ALTER TABLE nexa_threads DROP CONSTRAINT IF EXISTS nexa_threads_agent_mode_check;
UPDATE nexa_threads SET agent_mode = 'flex' WHERE agent_mode = 'agentic';
ALTER TABLE nexa_threads ADD CONSTRAINT nexa_threads_agent_mode_check CHECK (agent_mode IN ('direct', 'flex'));
