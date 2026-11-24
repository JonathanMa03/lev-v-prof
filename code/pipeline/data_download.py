"""Reproducible SEC manufacturing universe and annual financial extraction.

Python is used for data engineering only. Modeling is performed in R.

The current SEC ticker universe is a snapshot, NOT a historical 2024 listing panel.
The SEC 'fy' field alone does not establish the period of a reported fact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
US_STATES = frozenset('AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC'.split())
EXCHANGE_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'
SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'


def default_tags():
    return {
        'assets': ('Assets',),
        'liabilities': ('Liabilities',),
        'liabilities_current': ('LiabilitiesCurrent',),
        'liabilities_noncurrent': ('LiabilitiesNoncurrent',),
        'temporary_equity': ('TemporaryEquityCarryingAmountAttributableToParent', 'TemporaryEquityCarryingAmount'),
        'equity_total': ('StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',),
        'equity_stockholders': ('StockholdersEquity',),
        'minority_interest': ('MinorityInterest',),
        'net_income': ('NetIncomeLoss', 'ProfitLoss'),
        'revenue': ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'),
    }


@dataclass
class DownloadConfig:
    fiscal_year: int = 2024
    sic_min: int = 2000
    sic_max: int = 3999
    exchanges: tuple[str, ...] = ('Nasdaq', 'NYSE', 'CBOE', 'TXSE')
    require_us_incorporation: bool = True
    include_unknown_incorporation: bool = False
    forms: tuple[str, ...] = ('10-K',)
    # Strict calendar-end policy reproduces the feasibility notebook's rule.
    fiscal_end_policy: str = 'calendar_year'  # or 'fy_only'
    annual_days_min: int = 330
    annual_days_max: int = 380
    identity_tolerance: float = 0.001
    user_agent: str = 'Researcher Real Name real-email@institution.edu'
    request_delay: float = 0.25
    timeout: float = 45
    max_retries: int = 4
    backoff_seconds: float = 2.0
    cache_dir: Path = ROOT / 'data' / 'raw' / 'sec'
    output_dir: Path = ROOT / 'data' / 'processed'
    refresh_exchange: bool = False
    refresh_metadata: bool = False
    refresh_facts: bool = False
    offline: bool = False
    max_firms: int | None = None
    save_outputs: bool = True
    tags: dict[str, tuple[str, ...]] = field(default_factory=default_tags)

    def validate(self):
        if self.fiscal_end_policy not in ('calendar_year', 'fy_only'):
            raise ValueError('fiscal_end_policy must be calendar_year or fy_only')
        if self.sic_min > self.sic_max or self.max_retries < 1 or self.request_delay < 0:
            raise ValueError('Invalid SIC, retry, or delay configuration')
        if self.max_firms is not None and self.max_firms < 1:
            raise ValueError('max_firms must be positive or None')
        if not self.offline and ('@' not in self.user_agent or 'real-email@' in self.user_agent or 'example.com' in self.user_agent):
            raise ValueError('Set user_agent to a real name and working contact email before SEC requests')
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)


def cik10(value: Any) -> str:
    s = str(value).strip()
    if not s.isdigit() or len(s) > 10:
        raise ValueError(f'Invalid CIK: {value!r}')
    return s.zfill(10)


def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class SECClient:
    """Sequential, rate-limited SEC client with bounded exponential retries."""
    def __init__(self, cfg: DownloadConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': cfg.user_agent, 'Accept': 'application/json', 'Accept-Encoding': 'gzip, deflate'})
        self.last_request = 0.0

    def get(self, url: str) -> dict:
        if self.cfg.offline:
            raise RuntimeError(f'Offline and no usable cache: {url}')
        for attempt in range(self.cfg.max_retries):
            wait = self.cfg.request_delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                self.last_request = time.monotonic()
                r = self.session.get(url, timeout=self.cfg.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt == self.cfg.max_retries - 1:
                        r.raise_for_status()
                    retry_after = r.headers.get('Retry-After', '')
                    try:
                        delay = min(float(retry_after), 120) if retry_after else self.cfg.backoff_seconds * 2**attempt
                    except ValueError:
                        delay = self.cfg.backoff_seconds * 2**attempt
                    time.sleep(delay + random.uniform(0, 0.2))
                    continue
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise ValueError(f'Expected JSON object from {url}')
                return data
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.cfg.max_retries - 1:
                    raise
                time.sleep(self.cfg.backoff_seconds * 2**attempt)
        raise RuntimeError(f'Retries exhausted: {url}')

    def close(self):
        self.session.close()


def cached_api(client: SECClient, path: Path, url: str, valid, refresh=False) -> tuple[dict, str]:
    if not refresh:
        obj = read_json(path)
        if obj is not None and valid(obj):
            return obj, 'cached'
    obj = client.get(url)
    if not valid(obj):
        raise ValueError(f'Invalid response: {url}')
    atomic_json(path, obj)
    return obj, 'downloaded'


def parse_exchange_listing(obj: dict) -> pd.DataFrame:
    fields, rows = obj.get('fields'), obj.get('data')
    if not isinstance(fields, list) or not isinstance(rows, list) or not {'cik', 'exchange'}.issubset(fields):
        raise ValueError('Unexpected SEC exchange listing schema')
    frame = pd.DataFrame(rows, columns=fields)
    frame['cik_str'] = frame['cik'].map(cik10)
    result = frame.groupby('cik_str', as_index=False).agg(
        exchanges=('exchange', lambda x: sorted({str(v) for v in x.dropna()})),
        tickers=('ticker', lambda x: sorted({str(v) for v in x.dropna()})),
    )
    return result


def normalize_metadata(obj: dict, cik: str) -> dict:
    # Original SEC submissions use 'sic', 'stateOfIncorporation', etc.
    return {
        'cik_str': cik,
        'name': obj.get('name'),
        'sic': obj.get('sic'),
        'sic_description': obj.get('sicDescription'),
        'state_of_incorporation': obj.get('stateOfIncorporation'),
        'incorporation_description': obj.get('stateOfIncorporationDescription'),
        'state_of_business': (obj.get('addresses') or {}).get('business', {}).get('stateOrCountry'),
    }


def build_universe(cfg: DownloadConfig, client: SECClient) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build universe from cached SEC ticker snapshot and submissions metadata.

    All exchange-listed CIKs are audited. An existing metadata JSONL from
    feasibility is imported when present, but missing entries are fetched.
    """
    raw = cfg.cache_dir
    exchange_obj, exchange_status = cached_api(
        client, raw / 'company_tickers_exchange.json', EXCHANGE_URL,
        lambda x: isinstance(x.get('fields'), list) and isinstance(x.get('data'), list),
        refresh=cfg.refresh_exchange,
    )
    exchanges = parse_exchange_listing(exchange_obj)
    metadata_dir = raw / 'submissions'
    legacy_path = raw.parent / 'sec_company_metadata.jsonl'
    legacy = {}
    if legacy_path.exists() and not cfg.refresh_metadata:
        with legacy_path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    key = cik10(row.get('cik_str', row.get('cik')))
                    legacy[key] = row
                except (ValueError, TypeError):
                    continue
    records, logs = [], []
    for cik in tqdm(exchanges['cik_str'], desc='SEC submissions metadata'):
        path = metadata_dir / f'CIK{cik}.json'
        try:
            obj = read_json(path) if not cfg.refresh_metadata else None
            if obj is not None and cik10(obj.get('cik')) == cik:
                row, status = normalize_metadata(obj, cik), 'cached'
            elif cik in legacy and {'sic', 'state_of_incorporation'}.issubset(legacy[cik]):
                row, status = {**legacy[cik], 'cik_str': cik}, 'legacy_cached'
            else:
                obj, status = cached_api(client, path, SUBMISSIONS_URL.format(cik=cik),
                                         lambda x: str(x.get('cik', '')).isdigit() and cik10(x['cik']) == cik,
                                         refresh=cfg.refresh_metadata)
                row = normalize_metadata(obj, cik)
            records.append(row)
            logs.append({'cik_str': cik, 'stage': 'submissions', 'status': status, 'error': None})
        except (requests.RequestException, ValueError, RuntimeError, OSError) as e:
            logs.append({'cik_str': cik, 'stage': 'submissions', 'status': 'failed', 'error': str(e)})
    metadata = pd.DataFrame(records)
    if metadata.empty:
        raise RuntimeError('No submissions metadata available; check cache and network')
    metadata['cik_str'] = metadata['cik_str'].map(cik10)
    metadata = metadata.drop_duplicates('cik_str', keep='last')
    all_firms = exchanges.merge(metadata, on='cik_str', how='left', validate='one_to_one')
    all_firms['sic_numeric'] = pd.to_numeric(all_firms.get('sic'), errors='coerce')
    all_firms['manufacturing'] = all_firms['sic_numeric'].between(cfg.sic_min, cfg.sic_max)
    codes = all_firms['state_of_incorporation'].fillna('').astype(str).str.strip().str.upper()
    all_firms['incorporation_status'] = np.select(
        [codes.isin(US_STATES), codes.eq('')], ['confirmed_us', 'unknown'], default='non_us')
    all_firms['selected_exchange'] = all_firms['exchanges'].map(lambda x: bool(set(x) & set(cfg.exchanges)))
    incorporation_ok = (all_firms['incorporation_status'] == 'confirmed_us')
    if cfg.include_unknown_incorporation:
        incorporation_ok |= all_firms['incorporation_status'] == 'unknown'
    if not cfg.require_us_incorporation:
        incorporation_ok = pd.Series(True, index=all_firms.index)
    all_firms['eligible'] = all_firms['manufacturing'] & all_firms['selected_exchange'] & incorporation_ok
    all_firms['exclusion_reason'] = np.select(
        [~all_firms['manufacturing'], ~all_firms['selected_exchange'], ~incorporation_ok],
        ['outside_sic_or_missing_sic', 'exchange_not_selected', 'incorporation_not_eligible'],
        default='eligible')
    eligible = all_firms.loc[all_firms['eligible']].sort_values('cik_str').copy()
    if cfg.max_firms is not None:
        eligible = eligible.head(cfg.max_firms).copy()
    LOG.info('Exchange snapshot %s: %d rows; eligible %d', exchange_status, len(exchanges), len(eligible))
    return eligible, all_firms, pd.DataFrame(logs)


def facts_valid(obj: dict, cik: str) -> bool:
    return isinstance(obj, dict) and str(obj.get('cik', '')).isdigit() and cik10(obj['cik']) == cik and isinstance(obj.get('facts'), dict)


def download_facts(firms: pd.DataFrame, cfg: DownloadConfig, client: SECClient) -> pd.DataFrame:
    records = []
    for cik in tqdm(firms['cik_str'], desc='SEC Company Facts'):
        try:
            _, status = cached_api(client, cfg.cache_dir / 'companyfacts' / f'CIK{cik}.json',
                                   FACTS_URL.format(cik=cik), lambda x: facts_valid(x, cik),
                                   refresh=cfg.refresh_facts)
            records.append({'cik_str': cik, 'stage': 'companyfacts', 'status': status, 'error': None})
        except (requests.RequestException, ValueError, RuntimeError, OSError) as e:
            records.append({'cik_str': cik, 'stage': 'companyfacts', 'status': 'failed', 'error': str(e)})
    return pd.DataFrame(records)


def eligible_filing(obj: dict, cfg: DownloadConfig) -> dict | None:
    """Choose a filing using FY assets; keep same accession for all concepts.

    Explicitly require a FY-tagged fact and an end date. Beware that SEC fy
    may describe the filing context rather than the observation's own FY.
    """
    units = obj.get('facts', {}).get('us-gaap', {}).get('Assets', {}).get('units', {}).get('USD', [])
    candidates = [x for x in units if x.get('form') in cfg.forms and x.get('fy') == cfg.fiscal_year
                  and x.get('fp') == 'FY' and x.get('accn') and x.get('end') and not x.get('start')
                  and (cfg.fiscal_end_policy != 'calendar_year' or str(x['end']).startswith(str(cfg.fiscal_year)))]
    if not candidates:
        return None
    x = max(candidates, key=lambda r: (r.get('end', ''), r.get('filed', ''), r.get('accn', '')))
    return {'fiscal_end': x['end'], 'accession': x['accn'], 'filing_date': x.get('filed'), 'filing_form': x.get('form')}


def select_fact(gaap: dict, tags: tuple[str, ...], variable: str, filing: dict, cfg: DownloadConfig) -> tuple[float, str | None, str]:
    """Choose an exact accession/end match; flag conflicts rather than guessing."""
    balance = variable in {'assets', 'liabilities', 'liabilities_current', 'liabilities_noncurrent', 'temporary_equity', 'equity_total', 'equity_stockholders', 'minority_interest'}
    for tag in tags:
        observations = gaap.get(tag, {}).get('units', {}).get('USD', [])
        matches = []
        for x in observations:
            if x.get('accn') != filing['accession'] or x.get('end') != filing['fiscal_end'] or x.get('form') not in cfg.forms:
                continue
            if balance:
                if x.get('start') is not None:
                    continue
            else:
                if not x.get('start'):
                    continue
                try:
                    days = (pd.Timestamp(x['end']) - pd.Timestamp(x['start'])).days
                except (ValueError, TypeError):
                    continue
                if not cfg.annual_days_min <= days <= cfg.annual_days_max:
                    continue
            if isinstance(x.get('val'), (int, float)) and np.isfinite(x['val']):
                matches.append(x)
        if matches:
            # Different annual start dates or values in the same filing require review.
            keys = {(x.get('start'), float(x['val'])) for x in matches}
            if len(keys) > 1:
                return np.nan, tag, 'conflict'
            return float(matches[0]['val']), tag, 'ok'
    return np.nan, None, 'missing'


def extract_one(cik: str, obj: dict, cfg: DownloadConfig) -> dict:
    row = {'cik_str': cik, 'fiscal_year': cfg.fiscal_year, 'extraction_status': 'no_eligible_filing',
           'fiscal_end': None, 'accession': None, 'filing_date': None, 'filing_form': None}
    for key in cfg.tags:
        row.update({key: np.nan, key + '_tag': None, key + '_status': 'missing'})
    filing = eligible_filing(obj, cfg)
    if filing is None:
        return row
    row.update(filing)
    gaap = obj.get('facts', {}).get('us-gaap', {})
    for variable, tags in cfg.tags.items():
        val, tag, status = select_fact(gaap, tags, variable, filing, cfg)
        row.update({variable: val, variable + '_tag': tag, variable + '_status': status})
    row['extraction_status'] = 'extracted_with_conflict' if any(row[k + '_status'] == 'conflict' for k in cfg.tags) else 'extracted'
    return row


def construct_ratios(df: pd.DataFrame, cfg: DownloadConfig) -> pd.DataFrame:
    """Apply conservative liabilities hierarchy and preserve accounting provenance.

    Primary: reported total; current + noncurrent (only when consistent
    with independently available balance-sheet components); assets minus
    consolidated equity (only when no separate temporary equity is present).
    Stockholders-only imputation is kept as a separate sensitivity measure.
    """
    out = df.copy()
    required = set(default_tags()) | set(cfg.tags)
    for col in required:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce')

    assets = out['assets']
    valid_assets = assets.where(assets > 0)
    tol = cfg.identity_tolerance

    out['total_equity'] = out['equity_total'].copy()
    components = out['equity_stockholders'].notna() & out['minority_interest'].notna()
    use_components = out['total_equity'].isna() & components
    out.loc[use_components, 'total_equity'] = (
        out.loc[use_components, 'equity_stockholders']
        + out.loc[use_components, 'minority_interest']
    )
    out['equity_basis'] = np.select(
        [out['equity_total'].notna(), use_components],
        ['consolidated', 'stockholders_plus_minority'], default='unavailable'
    )

    # Do not infer missing minority interest to be zero.
    out['provisional_equity'] = out['total_equity'].fillna(out['equity_stockholders'])
    out['provisional_liabilities'] = assets - out['equity_stockholders']
    out['provisional_leverage'] = out['provisional_liabilities'].where(
        out['provisional_liabilities'] >= 0
    ) / valid_assets

    # Temporary/mezzanine equity is outside permanent stockholders' equity.
    # The SEC tag is not assumed to be zero if it is absent.
    out['temporary_equity_for_identity'] = out['temporary_equity'].fillna(0)
    out['liabilities_implied'] = (
        assets - out['provisional_equity'] - out['temporary_equity_for_identity']
    )
    out['identity_difference'] = out['liabilities'] - out['liabilities_implied']
    out['identity_relative_difference'] = out['identity_difference'].abs() / valid_assets
    out['identity_consistent'] = (
        out['identity_relative_difference'] <= tol
    ).astype('boolean')
    out.loc[out['identity_relative_difference'].isna(), 'identity_consistent'] = pd.NA

    # First-priority liabilities are the direct total-liabilities tag.
    out['liabilities_final'] = out['liabilities'].copy()
    out['liabilities_source'] = np.where(out['liabilities'].notna(), 'reported', 'missing')

    # Second priority: current + noncurrent totals, never individual liability
    # subaccounts. Check against assets less stockholders equity, allowing
    # separately tagged minority or temporary equity where available.
    component_sum = out['liabilities_current'] + out['liabilities_noncurrent']
    out['liabilities_component_sum'] = component_sum
    equity_check = out['total_equity'].fillna(out['equity_stockholders'])
    residual = assets - component_sum - equity_check - out['temporary_equity_for_identity']
    out['component_identity_relative_difference'] = residual.abs() / valid_assets
    out['component_identity_consistent'] = (
        out['component_identity_relative_difference'] <= tol
    ).astype('boolean')
    out.loc[out['component_identity_relative_difference'].isna(), 'component_identity_consistent'] = pd.NA
    component_ok = (
        out['liabilities_final'].isna()
        & component_sum.notna()
        & component_sum.ge(0)
        & out['component_identity_consistent'].fillna(False)
    )
    out.loc[component_ok, 'liabilities_final'] = component_sum.loc[component_ok]
    out.loc[component_ok, 'liabilities_source'] = 'current_plus_noncurrent'

    # Third priority: consolidated equity permits a residual calculation;
    # explicitly identified temporary equity must also be deducted.
    implied_consolidated = (
        assets - out['total_equity'] - out['temporary_equity_for_identity']
    )
    reconstructed = (
        out['liabilities_final'].isna()
        & valid_assets.notna()
        & out['total_equity'].notna()
        & implied_consolidated.ge(0)
    )
    out.loc[reconstructed, 'liabilities_final'] = implied_consolidated.loc[reconstructed]
    out.loc[reconstructed, 'liabilities_source'] = 'reconstructed_consolidated_equity'

    provisional = (
        out['liabilities_final'].isna()
        & out['total_equity'].isna()
        & out['equity_stockholders'].notna()
    )
    out.loc[provisional, 'liabilities_source'] = 'provisional_equity_only'

    out['roa'] = out['net_income'] / valid_assets
    out['leverage'] = out['liabilities_final'].where(
        out['liabilities_final'] >= 0
    ) / valid_assets

    # Retain reported liabilities even when a partial-equity identity fails.
    # Flag for review rather than overwrite or silently drop.
    out['accounting_review_flag'] = (
        (out['liabilities_source'] == 'reported')
        & (out['identity_consistent'] == False).fillna(False)
    )
    out['has_core_variables'] = out['roa'].notna() & out['leverage'].notna()
    return out


def extract_facts(firms: pd.DataFrame, cfg: DownloadConfig) -> pd.DataFrame:
    records = []
    for cik in tqdm(firms['cik_str'], desc='Extracting financials'):
        obj = read_json(cfg.cache_dir / 'companyfacts' / f'CIK{cik}.json')
        if obj is None or not facts_valid(obj, cik):
            records.append({'cik_str': cik, 'extraction_status': 'missing_or_invalid_cache'})
            continue
        try:
            records.append(extract_one(cik, obj, cfg))
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            LOG.exception('Financial extraction failed for %s', cik)
            records.append({'cik_str': cik, 'extraction_status': 'error', 'extraction_error': str(e)})
    extracted = pd.DataFrame(records)
    return construct_ratios(firms.merge(extracted, on='cik_str', how='left', validate='one_to_one'), cfg)


def save_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (list, dict))).any():
            out[col] = out[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return out


def run_pipeline(config: DownloadConfig | None = None) -> dict[str, pd.DataFrame]:
    """End-to-end reproducible pipeline; returns universe, financials, logs, audit.

    `offline=True` forbids network requests. Cache files are never deleted.
    `max_firms` limits financial downloads, not universe construction.
    """
    cfg = config or DownloadConfig()
    cfg.validate()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    client = SECClient(cfg)
    try:
        universe, audit, metadata_log = build_universe(cfg, client)
        download_log = download_facts(universe, cfg, client)
    finally:
        client.close()
    financials = extract_facts(universe, cfg)
    if cfg.save_outputs:
        year = cfg.fiscal_year
        save_csv(cfg.output_dir / 'eligible_manufacturing_firms.csv', csv_safe(universe))
        save_csv(cfg.output_dir / 'manufacturing_universe_audit.csv', csv_safe(audit))
        save_csv(cfg.output_dir / f'manufacturing_financials_{year}_full.csv', csv_safe(financials))
        save_csv(cfg.output_dir / f'manufacturing_financials_{year}_analysis.csv', csv_safe(financials.loc[financials['has_core_variables']]))
        save_csv(cfg.output_dir / f'sec_download_log_{year}.csv', download_log)
        save_csv(cfg.output_dir / f'sec_metadata_log_{year}.csv', metadata_log)
        manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                    'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                    'universe_count': len(universe), 'analysis_count': int(financials['has_core_variables'].sum()),
                    'note': 'Current SEC exchange/metadata snapshot; not a historical listing census.'}
        atomic_json(cfg.output_dir / f'pipeline_manifest_{year}.json', manifest)
    return {'universe': universe, 'financials': financials, 'download_log': download_log,
            'metadata_log': metadata_log, 'universe_audit': audit}
