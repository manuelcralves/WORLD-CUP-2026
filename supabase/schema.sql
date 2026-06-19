-- Beat the Machine · Supabase schema
-- Run once: Supabase dashboard → SQL Editor → New query → paste → Run.
-- Tables: profiles (per user), matches (fed by the daily pipeline), predictions
-- (written by users, locked at kickoff via RLS). A leaderboard view ranks the
-- humans plus "The Machine" (the model's own picks). Anonymous (guest) auth is
-- supported — a guest is a real auth user, so the same policies apply.

-- ----------------------------------------------------------------- profiles --
create table if not exists public.profiles (
  id         uuid primary key references auth.users(id) on delete cascade,
  name       text,
  avatar     text,
  created_at timestamptz default now()
);

-- create a profile automatically on sign-up (Google name/avatar, or "Player")
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, name, avatar) values (
    new.id,
    coalesce(new.raw_user_meta_data->>'full_name',
             new.raw_user_meta_data->>'name', 'Player'),
    new.raw_user_meta_data->>'avatar_url'
  ) on conflict (id) do nothing;
  return new;
end; $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users for each row
  execute function public.handle_new_user();

-- ------------------------------------------------------------------ matches --
-- Upserted by the daily GitHub Action (service-role key). match_id = "home|away".
create table if not exists public.matches (
  match_id   text primary key,
  home       text not null,
  away       text not null,
  kickoff    timestamptz,
  home_score int,
  away_score int,
  model_home int,
  model_away int,
  stage      text,
  played     boolean default false,
  advance    text,        -- knockout: who advanced (penalty-shootout winner for a level tie)
  updated_at timestamptz default now()
);
alter table public.matches add column if not exists advance text;   -- for tables created before this column

-- -------------------------------------------------------------- predictions --
create table if not exists public.predictions (
  user_id    uuid not null references auth.users(id) on delete cascade,
  match_id   text not null references public.matches(match_id) on delete cascade,
  pred_home  int  not null check (pred_home between 0 and 30),
  pred_away  int  not null check (pred_away between 0 and 30),
  updated_at timestamptz default now(),
  primary key (user_id, match_id)
);

-- --- scoring (cumulative, FIFA-style): result +10, each team's goals +5, ---------
-- goal-difference +5, exact-score bonus +5. A perfect score = 10+5+5+5+5 = 30.
create or replace function public.pts(ph int, pa int, ah int, aa int)
returns int language sql immutable as $$
  select case when ah is null or ph is null then 0 else
      (case when (ph > pa) = (ah > aa) and (ph < pa) = (ah < aa) then 10 else 0 end)
    + (case when ph = ah then 5 else 0 end)
    + (case when pa = aa then 5 else 0 end)
    + (case when (ph - pa) = (ah - aa) then 5 else 0 end)
    + (case when ph = ah and pa = aa then 5 else 0 end)
  end;
$$;

-- ----------------------------------------- leaderboard (humans + Machine) ----
create or replace view public.leaderboard as
  select uid, name, avatar, points, played, is_model, picks from (
    select pr.id::text as uid, coalesce(pr.name, 'Player') as name, pr.avatar,
           coalesce(sum(public.pts(pd.pred_home, pd.pred_away,
                                   m.home_score, m.away_score)), 0)::int as points,
           count(m.match_id) filter (where m.played)::int as played,
           false as is_model,
           count(m.match_id)::int as picks
    from public.profiles pr
    left join public.predictions pd on pd.user_id = pr.id
    left join public.matches m       on m.match_id = pd.match_id
                                    and m.kickoff >= '2026-06-18 16:00:00+00'::timestamptz
    group by pr.id, pr.name, pr.avatar
    union all
    select 'machine', '🤖 The Machine', null::text,
           coalesce(sum(public.pts(model_home, model_away,
                                   home_score, away_score)), 0)::int,
           count(*) filter (where played)::int, true, count(*)::int
    from public.matches
    where kickoff >= '2026-06-18 16:00:00+00'::timestamptz   -- fresh start: from Czech Republic vs South Africa onward
  ) t order by points desc;

-- ------------------------------------------------------------------- RLS -----
alter table public.profiles    enable row level security;
alter table public.matches     enable row level security;
alter table public.predictions enable row level security;

create policy "profiles readable" on public.profiles
  for select using (true);
create policy "profiles own" on public.profiles
  for all using (auth.uid() = id) with check (auth.uid() = id);

create policy "matches readable" on public.matches
  for select using (true);                       -- only the service key writes

-- read your own predictions any time; other players' only once the match has
-- kicked off (so nobody can copy upcoming picks)
create policy "preds read own or post-kickoff" on public.predictions
  for select using (
    auth.uid() = user_id
    or now() >= coalesce(
      (select kickoff from public.matches m where m.match_id = predictions.match_id),
      'infinity'::timestamptz));
create policy "preds insert before kickoff" on public.predictions
  for insert to authenticated with check (
    auth.uid() = user_id and now() < coalesce(
      (select kickoff from public.matches m where m.match_id = predictions.match_id),
      'infinity'::timestamptz));
create policy "preds update before kickoff" on public.predictions
  for update to authenticated using (auth.uid() = user_id) with check (
    auth.uid() = user_id and now() < coalesce(
      (select kickoff from public.matches m where m.match_id = predictions.match_id),
      'infinity'::timestamptz));

grant select on public.leaderboard to anon, authenticated;

-- ------------------------------------------ crowd W/D/L split per match ------
-- Aggregate only (counts/percentages) — never exposes an individual pick, so
-- it's safe to read for upcoming matches too.
create or replace view public.match_crowd as
  select match_id, count(*)::int as n,
         round(100.0 * count(*) filter (where pred_home > pred_away) / count(*))::int as home_pct,
         round(100.0 * count(*) filter (where pred_home = pred_away) / count(*))::int as draw_pct,
         round(100.0 * count(*) filter (where pred_home < pred_away) / count(*))::int as away_pct
  from public.predictions
  group by match_id;

grant select on public.match_crowd to anon, authenticated;

-- ----------------------------------------------------------------- brackets --
-- Knockout-bracket predictions: one row per user, the whole bracket as JSON
-- ({match_id: picked_team}). Opens once the group stage is complete and locks
-- at the first Round-of-32 kickoff. Scored separately from the match game.
create table if not exists public.brackets (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  picks      jsonb not null default '{}'::jsonb,
  updated_at timestamptz default now()
);
alter table public.brackets enable row level security;
-- LOCK = the first Round-of-32 kickoff: 2026-06-28 20:00 WEST (19:00 UTC).
drop policy if exists "brackets own" on public.brackets;
-- read your own bracket any time; everyone else's only AFTER the lock (anti-copy):
create policy "brackets read" on public.brackets for select using (
  auth.uid() = user_id
  or now() >= '2026-06-28 20:00:00+01'::timestamptz);
-- write your own, and only BEFORE the lock (no edits once the knockouts start):
create policy "brackets insert" on public.brackets for insert to authenticated
  with check (auth.uid() = user_id and now() < '2026-06-28 20:00:00+01'::timestamptz);
create policy "brackets update" on public.brackets for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id and now() < '2026-06-28 20:00:00+01'::timestamptz);
