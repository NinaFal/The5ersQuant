#!/usr/bin/env python3
"""
Download H1 data for ALL 37 assets from 2022-2025

Uses:
- OANDA API for forex pairs and metals
- Yahoo Finance for indices and crypto

Runs in background without disturbing the validation process.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# Load environment - use forexcom_demo which has OANDA credentials
load_dotenv('.env.forexcom_demo')

# Use all available cores
MAX_WORKERS = cpu_count()

# Configuration
DATA_DIR = Path('data/ohlcv')
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '2022-01-01'
END_DATE = '2025-12-31'

# All 37 tradable assets
ALL_FOREX = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'USD_CAD', 'AUD_USD', 'NZD_USD',
    'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
    'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
    'AUD_JPY', 'AUD_CHF', 'AUD_CAD', 'AUD_NZD',
    'NZD_JPY', 'NZD_CHF', 'NZD_CAD',
    'CHF_JPY', 'CAD_JPY', 'CAD_CHF',
]

ALL_METALS = ['XAU_USD', 'XAG_USD']
ALL_INDICES = ['SPX500_USD', 'NAS100_USD', 'UK100_USD']
ALL_CRYPTO = ['BTC_USD', 'ETH_USD']

OANDA_ASSETS = ALL_FOREX + ALL_METALS
YAHOO_ASSETS = ALL_INDICES + ALL_CRYPTO

print(f"=" * 80)
print(f"H1 DATA DOWNLOAD: 2022-2025 (ALL 37 ASSETS)")
print(f"=" * 80)
print(f"OANDA API: {len(OANDA_ASSETS)} assets")
print(f"Yahoo Finance: {len(YAHOO_ASSETS)} assets")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# OANDA API DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════

class OandaDownloader:
    """Download H1 data from OANDA API."""
    
    def __init__(self):
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.api_key = os.getenv('OANDA_API_KEY')
        self.environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
        
        if not self.account_id or not self.api_key:
            print("⚠️  OANDA credentials not found in .env")
            print("   Set OANDA_ACCOUNT_ID and OANDA_API_KEY")
            self.available = False
        else:
            self.available = True
        
        # Import oandapyV20
        try:
            import oandapyV20
            from oandapyV20.endpoints.instruments import InstrumentsCandles
            self.oandapyV20 = oandapyV20
            self.InstrumentsCandles = InstrumentsCandles
            
            # Create client
            if self.available:
                self.client = oandapyV20.API(access_token=self.api_key, environment=self.environment)
        except ImportError:
            print("⚠️  oandapyV20 not installed: pip install oandapyV20")
            self.available = False
    
    def download(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download H1 data for a symbol in chunks (OANDA has 5000 candle limit)."""
        if not self.available:
            return None
        
        # Split into 6-month chunks to stay under 5000 candle limit
        # 6 months ≈ 180 days × 24 hours = 4320 candles
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current_start = start_dt
        
        while current_start < end_dt:
            # Download 6 months at a time
            current_end = min(current_start + timedelta(days=180), end_dt)
            
            params = {
                'granularity': 'H1',
                'from': current_start.strftime('%Y-%m-%dT00:00:00Z'),
                'to': current_end.strftime('%Y-%m-%dT23:59:59Z'),
            }
            
            try:
                request = self.InstrumentsCandles(instrument=symbol, params=params)
                response = self.client.request(request)
                candles = response.get('candles', [])
                
                for candle in candles:
                    if candle['complete']:
                        all_data.append({
                            'time': candle['time'][:19],
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': int(candle['volume']),
                        })
                
                time.sleep(0.1)  # Rate limiting between chunks
                
            except Exception as e:
                print(f"   ⚠️  Chunk error {current_start.date()}: {e}")
            
            current_start = current_end
        
        if all_data:
            df = pd.DataFrame(all_data)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
            return df
        
        return None


# ═══════════════════════════════════════════════════════════════════════════
# YAHOO FINANCE DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════

class YahooDownloader:
    """Download H1 data from Yahoo Finance."""
    
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            print("⚠️  yfinance not installed: pip install yfinance")
            self.available = False
    
    def download(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download H1 data for a symbol."""
        if not self.available:
            return None
        
        # Map internal symbols to Yahoo tickers
        ticker_map = {
            'SPX500_USD': '^GSPC',   # S&P 500
            'NAS100_USD': '^IXIC',   # Nasdaq
            'UK100_USD': '^FTSE',    # FTSE 100
            'BTC_USD': 'BTC-USD',    # Bitcoin
            'ETH_USD': 'ETH-USD',    # Ethereum
        }
        
        ticker = ticker_map.get(symbol, symbol)
        
        try:
            # Yahoo doesn't have H1 for indices, download D1 instead
            df = self.yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1h',  # Try hourly first
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                # Fallback to daily
                df = self.yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    progress=False,
                    auto_adjust=True
                )
            
            if not df.empty:
                df = df.reset_index()
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                df['time'] = pd.to_datetime(df['time'])
                return df
            return None
            
        except Exception as e:
            print(f"   ❌ Yahoo error for {symbol}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def download_single_asset(symbol: str, downloader, asset_type: str) -> Dict:
    """Download single asset (for parallel execution)."""
    output_file = DATA_DIR / f"{symbol}_H1.csv"
    
    # Skip if exists
    if output_file.exists():
        return {'symbol': symbol, 'status': 'skipped'}
    
    try:
        df = downloader.download(symbol, START_DATE, END_DATE)
        
        if df is not None and len(df) > 0:
            df.to_csv(output_file, index=False)
            return {'symbol': symbol, 'status': 'success', 'rows': len(df)}
        else:
            return {'symbol': symbol, 'status': 'failed', 'error': 'No data returned'}
    except Exception as e:
        return {'symbol': symbol, 'status': 'failed', 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def download_all_h1_data():
    """Download H1 data for all assets using parallel processing."""
    
    oanda = OandaDownloader()
    yahoo = YahooDownloader()
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    print(f"\n🚀 Using {MAX_WORKERS} CPU cores for parallel downloads\n")
    
    # Download OANDA assets in parallel
    print(f"📊 Downloading OANDA assets ({len(OANDA_ASSETS)}) in parallel...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_asset, symbol, oanda, 'OANDA'): symbol 
            for symbol in OANDA_ASSETS
        }
        
        with tqdm(total=len(OANDA_ASSETS), desc="OANDA") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['status'] == 'success':
                    results['success'].append(result['symbol'])
                elif result['status'] == 'skipped':
                    results['skipped'].append(result['symbol'])
                else:
                    results['failed'].append(result['symbol'])
                
                pbar.update(1)
    
    # Download Yahoo assets in parallel
    print(f"\n📊 Downloading Yahoo Finance assets ({len(YAHOO_ASSETS)}) in parallel...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_asset, symbol, yahoo, 'Yahoo'): symbol 
            for symbol in YAHOO_ASSETS
        }
        
        with tqdm(total=len(YAHOO_ASSETS), desc="Yahoo") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result['status'] == 'success':
                    results['success'].append(result['symbol'])
                elif result['status'] == 'skipped':
                    results['skipped'].append(result['symbol'])
                else:
                    results['failed'].append(result['symbol'])
                
                pbar.update(1)
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"DOWNLOAD SUMMARY")
    print(f"=" * 80)
    print(f"✅ Success: {len(results['success'])} assets")
    print(f"⏭️  Skipped: {len(results['skipped'])} assets (already exist)")
    print(f"❌ Failed:  {len(results['failed'])} assets")
    
    if results['success']:
        print(f"\nSuccessfully downloaded:")
        for s in results['success']:
            print(f"  - {s}")
    
    if results['failed']:
        print(f"\nFailed to download:")
        for s in results['failed']:
            print(f"  - {s}")
    
    # Save log
    log_file = Path('analysis/h1_download_log.txt')
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, 'w') as f:
        f.write(f"H1 Download Log - {datetime.now()}\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n\n")
        f.write(f"Success: {len(results['success'])}\n")
        f.write(f"Skipped: {len(results['skipped'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n\n")
        f.write(f"Details:\n")
        for s in results['success']:
            f.write(f"✅ {s}\n")
        for s in results['skipped']:
            f.write(f"⏭️  {s}\n")
        for s in results['failed']:
            f.write(f"❌ {s}\n")
    
    print(f"\n📝 Log saved to: {log_file}")
    print(f"=" * 80)


if __name__ == '__main__':
    download_all_h1_data()
