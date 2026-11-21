-- AgentLab tables for the optional SupabaseStore (Postgres via Supabase's REST API).
-- Run this once in the Supabase SQL editor, then set:
--   AGENTLAB_STORE=supabase  SUPABASE_URL=https://<ref>.supabase.co  SUPABASE_KEY=<service-role key>
--
-- Security: row level security is enabled with NO policies, so the anon key can read or write nothing.
-- Use the service-role key (server side only, never in a browser or in git) or add your own policies.

create table if not exists agentlab_runs (
  run_id   text primary key,
  task     text,
  provider text,
  model    text,
  status   text,
  started  double precision,
  ended    double precision,
  meta     jsonb
);

create table if not exists agentlab_spans (
  span_id   text primary key,
  run_id    text not null,
  parent_id text,
  kind      text,
  name      text,
  input     jsonb,
  output    jsonb,
  status    text,
  started   double precision,
  ended     double precision,
  attrs     jsonb,
  seq       integer
);
create index if not exists idx_agentlab_spans_run on agentlab_spans (run_id, seq);

create table if not exists agentlab_checkpoints (
  run_id text    not null,
  seq    integer not null,
  state  jsonb,
  primary key (run_id, seq)
);

create table if not exists agentlab_eval_runs (
  eval_id text primary key,
  created double precision,
  suite   text,
  meta    jsonb
);

create table if not exists agentlab_eval_results (
  id      bigint generated always as identity primary key,  -- insertion order = report order
  eval_id text not null,
  config  text,
  task_id text,
  trial   integer,
  run_id  text,
  passed  integer,
  score   double precision,
  metrics jsonb,
  detail  text,
  status  text,
  answer  text
);
create index if not exists idx_agentlab_eval_results_eval on agentlab_eval_results (eval_id);

alter table agentlab_runs         enable row level security;
alter table agentlab_spans        enable row level security;
alter table agentlab_checkpoints  enable row level security;
alter table agentlab_eval_runs    enable row level security;
alter table agentlab_eval_results enable row level security;
