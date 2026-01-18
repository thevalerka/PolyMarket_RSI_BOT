#!/usr/bin/env python3
"""
Bot D2 Production - EMA Bands Trading with Polymarket Integration
Clean implementation with proper time-based sampling:
- Indicators (RSI/EMA): 1-second sampling
- Trading loop: 0.1-second for responsiveness
- Choppiness filter: Only trade when choppiness >= 30

pm2 start bot_d2_production_MAKER.py --cron-restart="00 * * * *" --interpreter python3
"""

import json
import time
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict
from collections import deque
from dataclasses import dataclass
import numpy as np
import logging

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_d2_production_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_d2_production_trades"

# Loop intervals
CHECK_INTERVAL = 0.1  # 100ms - trading loop
SAMPLE_INTERVAL = 1.0  # 1 second - indicator sampling

# Trading Parameters
MIN_BUY_PRICE = 0.05  # Minimum price to open position
MAX_BUY_PRICE = 0.95
CONFIRMATION_SECONDS = 0
START_DELAY = 20  # Wait 20s from period start
BUFFER_SECONDS = 20  # No trading in last 20s of period
POSITION_SIZE = 5.2  # 5 shares
MIN_SECONDS_BETWEEN_POSITIONS = 2
MIN_CHOPPINESS = 20  # Only trade when choppiness >= 30

# RSI/EMA Parameters
RSI_PERIOD_INITIAL = 60  # 60 seconds
RSI_BREACH_UPPER = 80
RSI_BREACH_LOWER = 20
RSI_PERIOD_MIN = 15
RSI_PERIOD_MAX = 120  # Cap at 120 seconds
RSI_PERIOD_ADJUSTMENT = 5
BREACH_WINDOW = 900  # 15 minutes

# Trading Thresholds
RSI_EXIT_CALL = 90
RSI_EXIT_PUT = 10
MIN_TOLERANCE = 5.0

# Persistence
STATE_SAVE_INTERVAL = 60

# ============================================================================
# POSITION DATACLASS
# ============================================================================

@dataclass
class Position:
    """Open position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float
    strike_price: float
    rsi_at_entry: float
    ema_at_entry: float
    tolerance: float
    volatility: float
    choppiness: float
    rsi_period: int
    max_profit: float = 0.0
    max_loss: float = 0.0
    time_to_expiry_at_entry: int = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def get_btc_price():
    """Get current BTC price from local JSON file"""
    try:
        data = read_json(BTC_FILE)
        if data:
            return data.get('price')
        return None
    except:
        return None

def get_strike_price() -> Optional[float]:
    """Get strike price from Bybit API"""
    try:
        import requests
        now = datetime.now(timezone.utc)
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                start_timestamp = int(period_start.timestamp() * 1000)

                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {
                    'category': 'linear',
                    'symbol': 'BTCUSDT',
                    'interval': '15',
                    'start': start_timestamp,
                    'limit': 1
                }

                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        kline_list = data.get('result', {}).get('list', [])
                        if kline_list:
                            return float(kline_list[0][1])
        return None
    except:
        return None

def get_seconds_to_expiry() -> int:
    """Calculate seconds until next 15-minute mark"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    seconds_into_quarter = minutes_into_quarter * 60 + now.second
    return 900 - seconds_into_quarter

def get_seconds_into_period() -> int:
    """Get seconds into current 15-minute period"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    return minutes_into_quarter * 60 + now.second

def get_bin_key(timestamp: float) -> str:
    """Get bin key for current period"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    minute = dt.minute

    for start_min in [0, 15, 30, 45]:
        if minute >= start_min and minute < start_min + 15:
            return f"{hour:02d}:{start_min:02d}"
    return f"{hour:02d}:00"

def calculate_rsi(prices: deque, period: int) -> Optional[float]:
    """Calculate RSI using proper Wilder's smoothing method"""
    if len(prices) < period + 1:
        return None

    # Get last period+1 prices to calculate period deltas
    prices_array = np.array(list(prices)[-(period + 1):])
    deltas = np.diff(prices_array)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # First average: Simple Moving Average of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing for remaining values (if any)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Handle division by zero
    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices: deque, period: int) -> Optional[float]:
    """Calculate EMA on 1-second samples"""
    if len(prices) < period:
        return None

    prices_array = np.array(list(prices))[-period:]
    k = 2.0 / (period + 1)
    ema = prices_array[0]

    for price in prices_array[1:]:
        ema = price * k + ema * (1 - k)

    return ema

def calculate_volatility(prices: deque, window: int = 60) -> float:
    """Calculate exponentially weighted volatility"""
    if len(prices) < 10:
        return 0.0

    prices_array = np.array(list(prices))
    weights = np.exp(np.linspace(-2, 0, len(prices_array)))
    weights = weights / weights.sum()

    weighted_mean = np.average(prices_array, weights=weights)
    variance = np.average((prices_array - weighted_mean) ** 2, weights=weights)

    return np.sqrt(variance)

def calculate_choppiness_index(prices: deque, period: int = 900) -> float:
    """
    Calculate Choppiness Index over the 15-minute period (900 seconds)

    Choppiness Index measures market choppiness on a scale of 0-100:
    - Values near 100 = Very choppy/ranging market (frequent direction changes)
    - Values near 0 = Strong trending market (consistent direction)
    - Values 38.2-61.8 = Transitional/neutral

    Formula: 100 * log10(sum(TR) / (max(high) - min(low))) / log10(period)
    Where TR = True Range = absolute price change for 1-second data

    Args:
        prices: Deque of 1-second prices
        period: Lookback period in seconds (default 900 = 15 minutes)

    Returns:
        Choppiness Index value (0-100)
    """
    if len(prices) < period:
        return 50.0  # Return neutral value if not enough data

    # Get last 'period' prices
    prices_array = np.array(list(prices)[-period:])

    # Calculate True Range for each period
    # Since we have 1-second prices, TR is just the absolute price change
    true_ranges = np.abs(np.diff(prices_array))
    sum_tr = np.sum(true_ranges)

    # Get high and low over the period
    high = np.max(prices_array)
    low = np.min(prices_array)
    high_low_range = high - low

    # Avoid division by zero
    if high_low_range == 0:
        return 100.0  # Completely flat = maximum choppiness

    # Calculate Choppiness Index
    if sum_tr == 0:
        return 100.0

    choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)

    # Clamp to 0-100 range
    choppiness = max(0.0, min(100.0, choppiness))

    return choppiness

# ============================================================================
# BOT CLASS
# ============================================================================

class BotD2Production:
    """Production trading bot with Polymarket integration"""

    def __init__(self, credentials: dict):
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking
        self.position: Optional[Position] = None
        self.last_position_close_time = 0

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Price history for indicators (1-second samples)
        self.price_history_1s = deque(maxlen=1800)  # 30 minutes

        # Period tracking
        self.strike_price: Optional[float] = None
        self.current_period_start: Optional[int] = None
        self.current_bin: Optional[str] = None

        # Buffer tracking
        self.start_buffer_reload_done = False

        # Last valid prices (for period-end closure)
        self.last_call_bid: Optional[float] = None
        self.last_put_bid: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_put_ask: Optional[float] = None
        self.last_btc_price: Optional[float] = None

        # Position verification
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()

        # Cached USDC balance (refresh every 10s)
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0

        # RSI/EMA state
        self.load_state()

        # Signal tracking
        self.last_signal_time = None
        self.last_signal_type = None

        # Adjustment tracking
        self.last_adjustment_time = time.time()
        self.total_breaches = 0

        # State save tracking
        self.last_state_save = time.time()

        # MAKER order tracking - BUY orders
        self.pending_buy_order_id: Optional[str] = None
        self.pending_buy_order_time: Optional[float] = None
        self.pending_buy_token_id: Optional[str] = None
        self.pending_buy_token_type: Optional[str] = None
        self.pending_buy_limit_price: Optional[float] = None
        self.tracked_bid_price: Optional[float] = None
        self.pending_buy_details: Optional[Dict] = None

        # MAKER order tracking - SELL orders
        self.pending_sell_order_id: Optional[str] = None
        self.pending_sell_order_time: Optional[float] = None
        self.pending_sell_token_id: Optional[str] = None
        self.pending_sell_limit_price: Optional[float] = None
        self.tracked_ask_price: Optional[float] = None
        self.pending_sell_details: Optional[Dict] = None

        logger.info("="*80)
        logger.info("🤖 BOT D2 PRODUCTION - EMA BANDS STRATEGY")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Min Choppiness: {MIN_CHOPPINESS}")
        logger.info(f"RSI Period: {self.rsi_period}s")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s")
        logger.info("="*80)

    def load_state(self):
        """Load bot state from persistent file"""
        state = read_json(STATE_FILE)
        if state:
            self.rsi_period = state.get('rsi_period', RSI_PERIOD_INITIAL)
            saved_history = state.get('price_history_1s', [])
            if saved_history:
                self.price_history_1s = deque(saved_history, maxlen=1800)
            else:
                self.rsi_period = RSI_PERIOD_INITIAL
            self.total_breaches = state.get('total_breaches', 0)
            self.last_adjustment_time = state.get('last_adjustment_time', time.time())
            logger.info(f"📥 Loaded state: RSI Period={self.rsi_period}s, History={len(self.price_history_1s)} samples")
        else:
            self.rsi_period = RSI_PERIOD_INITIAL
            logger.info(f"📥 No saved state, starting fresh: RSI Period={self.rsi_period}s")

    def save_state(self):
        """Save bot state to persistent file"""
        state = {
            'rsi_period': self.rsi_period,
            'price_history_1s': list(self.price_history_1s),
            'total_breaches': self.total_breaches,
            'last_adjustment_time': self.last_adjustment_time,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"d2_production_{today}.json"

    def load_today_trades(self):
        """Load today's trades if they exist"""
        filename = self.get_today_filename()
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.today_trades = data.get('trades', [])
                    logger.info(f"📂 Loaded {len(self.today_trades)} trades from {filename.name}")
            except:
                self.today_trades = []

    def save_trades(self):
        """Save today's trades to file"""
        filename = self.get_today_filename()

        daily_pnl = sum(t['pnl'] for t in self.today_trades)
        win_count = sum(1 for t in self.today_trades if t['pnl'] > 0)
        loss_count = sum(1 for t in self.today_trades if t['pnl'] < 0)

        data = {
            'date': date.today().isoformat(),
            'strategy': 'D2_EMA_Bands',
            'total_trades': len(self.today_trades),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_count / len(self.today_trades) if self.today_trades else 0,
            'daily_pnl': daily_pnl,
            'trades': self.today_trades
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def get_usdc_balance(self) -> float:
        """Get USDC balance from Polymarket"""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6

            return balance_usdc
        except Exception as e:
            logger.error(f"Error getting USDC balance: {e}")
            return 0.0

    def refresh_usdc_balance(self):
        """Refresh cached USDC balance if needed"""
        if time.time() - self.last_usdc_check >= self.usdc_check_interval:
            self.cached_usdc_balance = self.get_usdc_balance()
            self.last_usdc_check = time.time()
            logger.debug(f"💰 USDC balance refreshed: ${self.cached_usdc_balance:.2f}")

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files"""
        put_data = read_json(PUT_FILE)
        call_data = read_json(CALL_FILE)

        if put_data and call_data:
            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')

            put_changed = new_put_id != self.current_put_id
            call_changed = new_call_id != self.current_call_id

            if put_changed or call_changed:
                logger.info(f"   🔄 Asset IDs updated:")
                if put_changed:
                    logger.info(f"   PUT:  ...{new_put_id[-12:]}")
                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:]}")

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

            self.last_asset_reload = time.time()

    def verify_position_from_wallet(self):
        """Verify position matches wallet"""
        if not self.current_put_id or not self.current_call_id:
            return

        put_balance = self.check_token_balance(self.current_put_id)
        call_balance = self.check_token_balance(self.current_call_id)

        has_put = put_balance >= 0.5
        has_call = call_balance >= 0.5

        # Case 1: PUT position
        if has_put and not has_call:
            if self.position is None or self.position.token_type != 'PUT':
                logger.warning(f"⚠️  Wallet has PUT ({put_balance:.2f}), tracking: {self.position.token_type if self.position else 'None'}")
                # Sync position
                put_data = read_json(PUT_FILE)
                price = 0.50
                if put_data and put_data.get('best_bid'):
                    price = (put_data['best_bid'].get('price', 0.50) + put_data['best_ask'].get('price', 0.50)) / 2

                self.position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=put_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    rsi_at_entry=0,
                    ema_at_entry=0,
                    tolerance=0,
                    volatility=0,
                    choppiness=50,
                    rsi_period=self.rsi_period
                )

        # Case 2: CALL position
        elif has_call and not has_put:
            if self.position is None or self.position.token_type != 'CALL':
                logger.warning(f"⚠️  Wallet has CALL ({call_balance:.2f}), tracking: {self.position.token_type if self.position else 'None'}")
                # Sync position
                call_data = read_json(CALL_FILE)
                price = 0.50
                if call_data and call_data.get('best_bid'):
                    price = (call_data['best_bid'].get('price', 0.50) + call_data['best_ask'].get('price', 0.50)) / 2

                self.position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=call_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    rsi_at_entry=0,
                    ema_at_entry=0,
                    tolerance=0,
                    volatility=0,
                    choppiness=50,
                    rsi_period=self.rsi_period
                )

        # Case 3: No position
        elif not has_put and not has_call:
            if self.position is not None:
                logger.warning(f"⚠️  Wallet empty but tracking shows {self.position.token_type}")
                self.position = None

        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, bid_price: float,
                    btc_price: float, strike_price: float, rsi: float, ema: float,
                    tolerance: float, volatility: float, choppiness: float,
                    seconds_remaining: int, reason: str) -> bool:
        """Execute MAKER buy order - limit order $0.01 below bid"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING MAKER BUY ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Market Bid: ${bid_price:.4f}")

            # MAKER: Place limit order $0.01 below bid
            limit_price = max(0.01, bid_price - 0.01)
            logger.info(f"💰 Limit Price: ${limit_price:.4f} (bid - $0.01)")
            logger.info(f"📈 RSI: {rsi:.1f} (Period: {self.rsi_period}s)")
            logger.info(f"🌊 Choppiness: {choppiness:.1f}")
            logger.info(f"📝 Reason: {reason}")

            required = limit_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Use cached USDC balance
            cached_balance = self.cached_usdc_balance
            logger.info(f"💰 USDC Balance (cached): ${cached_balance:.2f}")

            MIN_BALANCE = 4.90
            if cached_balance < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${cached_balance:.2f} < ${MIN_BALANCE:.2f}")
                return False

            if cached_balance < required:
                logger.error(f"❌ Need: ${required:.2f}, Have: ${cached_balance:.2f}")
                return False

            # Store the bid price we're tracking
            self.tracked_bid_price = bid_price
            self.pending_buy_token_id = token_id
            self.pending_buy_token_type = token_type
            self.pending_buy_limit_price = limit_price

            # PHASE 1: Place limit order
            logger.info(f"\n🚀 PHASE 1: Placing MAKER limit buy order...")

            start_time = time.time()
            try:
                order_id = self.trader.place_buy_order(
                    token_id=token_id,
                    price=limit_price,
                    quantity=POSITION_SIZE
                )
            except Exception as order_error:
                logger.error(f"❌ Order error: {order_error}")
                self.tracked_bid_price = None
                self.pending_buy_token_id = None
                self.pending_buy_token_type = None
                self.pending_buy_limit_price = None
                return False

            if not order_id:
                logger.error(f"❌ Failed to place order")
                self.tracked_bid_price = None
                self.pending_buy_token_id = None
                self.pending_buy_token_type = None
                self.pending_buy_limit_price = None
                return False

            logger.info(f"✅ Limit order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")
            logger.info(f"⏳ Waiting for fill... (monitoring bid price)")

            self.pending_buy_order_id = order_id
            self.pending_buy_order_time = time.time()

            # Store order details for monitoring
            self.pending_buy_details = {
                'token_type': token_type,
                'token_id': token_id,
                'limit_price': limit_price,
                'btc_price': btc_price,
                'strike_price': strike_price,
                'rsi': rsi,
                'ema': ema,
                'tolerance': tolerance,
                'volatility': volatility,
                'choppiness': choppiness,
                'seconds_remaining': seconds_remaining
            }

            return True  # Order placed successfully, will monitor in main loop

        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            self.tracked_bid_price = None
            self.pending_buy_token_id = None
            self.pending_buy_token_type = None
            self.pending_buy_limit_price = None
            return False

    def execute_sell(self, token_id: str, ask_price: float, btc_price: float,
                     volatility: float, choppiness: float, seconds_remaining: int,
                     reason: str) -> bool:
        """Execute MAKER sell order - limit order $0.01 above ask"""
        try:
            # Verify we have tokens
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                logger.warning(f"\n⚠️  SELL ABORTED: No tokens in wallet")
                self.position = None
                return False

            size = actual_balance

            logger.info(f"\n{'='*60}")
            logger.info(f"💰 EXECUTING MAKER SELL ORDER - {reason}")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Size: {size:.2f} shares")
            logger.info(f"💰 Market Ask: ${ask_price:.4f}")

            # Calculate P&L (will be confirmed after fill)
            pnl = 0.0
            if self.position:
                # Estimate based on limit price
                limit_price = ask_price + 0.01
                pnl = (limit_price - self.position.entry_price) * size
                pnl_pct = ((limit_price / self.position.entry_price) - 1) * 100
                logger.info(f"📈 Est. P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # MAKER: Place limit order $0.01 above ask
            sell_price = ask_price + 0.01
            logger.info(f"💰 Limit Price: ${sell_price:.4f} (ask + $0.01)")

            # Store the ask price we're tracking
            self.tracked_ask_price = ask_price
            self.pending_sell_token_id = token_id
            self.pending_sell_limit_price = sell_price

            # Place order
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )

            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                self.tracked_ask_price = None
                self.pending_sell_token_id = None
                self.pending_sell_limit_price = None
                return False

            logger.info(f"✅ Limit sell order placed: {order_id[:16]}...")
            logger.info(f"⏳ Waiting for fill... (monitoring ask price)")

            self.pending_sell_order_id = order_id
            self.pending_sell_order_time = time.time()

            # Store details for final trade record
            self.pending_sell_details = {
                'btc_price': btc_price,
                'volatility': volatility,
                'choppiness': choppiness,
                'seconds_remaining': seconds_remaining,
                'reason': reason
            }

            return True  # Order placed successfully, will monitor in main loop

        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            self.tracked_ask_price = None
            self.pending_sell_token_id = None
            self.pending_sell_limit_price = None
            return False

    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        if self.position is not None:
            return False

        if time.time() - self.last_position_close_time < MIN_SECONDS_BETWEEN_POSITIONS:
            return False

        return True

    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t['pnl'] for t in self.today_trades)

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Bot D2 Production - EMA Bands Strategy\n")

        last_sample_time = 0

        try:
            while True:
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second
                current_time = time.time()
                timestamp = current_time

                # Determine current period
                period_start = None
                for start_min in [0, 15, 30, 45]:
                    if current_minute >= start_min and current_minute < start_min + 15:
                        period_start = start_min
                        break

                # Calculate time remaining
                if period_start is not None:
                    seconds_into_period = (current_minute - period_start) * 60 + current_second
                    seconds_remaining = 900 - seconds_into_period

                    # Buffer zones: first 20s and last 20s of period
                    in_start_buffer = seconds_into_period <= BUFFER_SECONDS
                    in_end_buffer = seconds_remaining <= BUFFER_SECONDS
                    in_buffer_zone = in_start_buffer or in_end_buffer
                else:
                    seconds_into_period = 0
                    seconds_remaining = 0
                    in_start_buffer = True
                    in_end_buffer = True
                    in_buffer_zone = True

                # Get bin key
                bin_key = get_bin_key(timestamp)

                # NEW PERIOD DETECTED
                if bin_key != self.current_bin:
                    # Close position from previous period
                    if self.position is not None and self.current_bin is not None:
                        logger.info(f"\n⚠️  PERIOD END - Closing position")

                        # Cancel any pending orders first
                        if self.pending_sell_order_id or self.pending_buy_order_id:
                            logger.info(f"🔄 Canceling pending orders...")
                            self.trader.cancel_all_orders()
                            self.pending_sell_order_id = None
                            self.pending_buy_order_id = None
                            self.pending_buy_token_id = None
                            self.pending_sell_token_id = None

                        # For MAKER strategy at period end, use last valid ask price
                        exit_price = self.last_call_ask if self.position.token_type == 'CALL' else self.last_put_ask
                        exit_btc = self.last_btc_price if self.last_btc_price else 0

                        if exit_price and exit_price > 0 and exit_btc > 0:
                            volatility = calculate_volatility(self.price_history_1s)
                            choppiness = calculate_choppiness_index(self.price_history_1s, period=900)

                            self.execute_sell(self.position.token_id, exit_price, exit_btc,
                                            volatility, choppiness, BUFFER_SECONDS - 1, "PERIOD_END")
                        else:
                            logger.error(f"❌ Cannot close - no valid exit price")

                    # Update period
                    self.current_bin = bin_key

                    # Reset buffer flag for new period
                    self.start_buffer_reload_done = False

                    # Reset last valid prices
                    self.last_call_bid = None
                    self.last_put_bid = None
                    self.last_call_ask = None
                    self.last_put_ask = None
                    self.last_btc_price = None

                    # Get strike price
                    self.strike_price = get_strike_price()

                    # Initial asset ID reload at period start
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start:02d})")
                    logger.info(f"✅ Strike: ${self.strike_price:.2f}" if self.strike_price else "❌ Strike: Not available")
                    logger.info(f"{'='*80}")
                    logger.info(f"⏳ Start buffer active: no trading for first {BUFFER_SECONDS}s")
                    self.reload_asset_ids()

                # Reload asset IDs at END of start buffer
                if period_start is not None and not self.start_buffer_reload_done:
                    if seconds_into_period > BUFFER_SECONDS and seconds_into_period <= BUFFER_SECONDS + 2:
                        logger.info(f"\n✅ START BUFFER ENDED - Final asset ID reload")
                        self.reload_asset_ids()
                        self.start_buffer_reload_done = True
                        logger.info(f"🟢 Trading now active for remaining period\n")

                # Reload asset IDs every 60s
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Verify position every 60s
                if time.time() - self.last_position_check >= 60:
                    self.verify_position_from_wallet()

                # Refresh cached USDC balance every 10s
                self.refresh_usdc_balance()

                # Get BTC price
                btc_price = get_btc_price()
                if btc_price is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Sample price every 1 second for indicators
                if current_time - last_sample_time >= SAMPLE_INTERVAL:
                    self.price_history_1s.append(btc_price)
                    last_sample_time = current_time

                    # Calculate indicators ONLY when new sample arrives (every 1 second)
                    rsi = calculate_rsi(self.price_history_1s, self.rsi_period)
                    ema = calculate_ema(self.price_history_1s, self.rsi_period)

                    # Calculate volatility
                    volatility = calculate_volatility(self.price_history_1s)

                    # Calculate Choppiness Index for the 15-minute period
                    choppiness = calculate_choppiness_index(self.price_history_1s, period=900)

                    # Count breaches and adjust RSI period every 10 seconds
                    if len(self.price_history_1s) >= self.rsi_period + 1:
                        if current_time - self.last_adjustment_time >= 10:
                            # Get last 15 minutes of data
                            breach_window_samples = min(BREACH_WINDOW, len(self.price_history_1s))
                            recent_prices = list(self.price_history_1s)[-breach_window_samples:]

                            breach_count = 0
                            in_breach = False

                            # Slide window of CURRENT rsi_period through data
                            num_windows = len(recent_prices) - self.rsi_period

                            if num_windows > 0:
                                for i in range(num_windows):
                                    window = recent_prices[i:i + self.rsi_period + 1]
                                    window_rsi = calculate_rsi(deque(window), self.rsi_period)

                                    if window_rsi is not None:
                                        is_outside = window_rsi > RSI_BREACH_UPPER or window_rsi < RSI_BREACH_LOWER

                                        # Count event START (transition from normal to breach)
                                        if is_outside and not in_breach:
                                            breach_count += 1
                                            in_breach = True
                                        # Track event END (transition from breach to normal)
                                        elif not is_outside and in_breach:
                                            in_breach = False

                            # Adjust period
                            old_period = self.rsi_period

                            if breach_count <= 3 and self.rsi_period > RSI_PERIOD_MIN:
                                self.rsi_period = max(RSI_PERIOD_MIN, self.rsi_period - RSI_PERIOD_ADJUSTMENT)
                                logger.info(f"[RSI ADJUST] Breaches: {breach_count} ≤ 3 → Period: {old_period}s → {self.rsi_period}s (DECREASE)")
                            elif breach_count >= 8 and self.rsi_period < RSI_PERIOD_MAX:
                                self.rsi_period = min(RSI_PERIOD_MAX, self.rsi_period + RSI_PERIOD_ADJUSTMENT)
                                logger.info(f"[RSI ADJUST] Breaches: {breach_count} ≥ 8 → Period: {old_period}s → {self.rsi_period}s (INCREASE)")
                            else:
                                logger.info(f"[RSI ADJUST] Breaches: {breach_count} (4-7) → Period: {old_period}s (NO CHANGE)")

                            self.last_adjustment_time = current_time
                            self.total_breaches = breach_count
                else:
                    # Use last calculated values
                    rsi = calculate_rsi(self.price_history_1s, self.rsi_period) if len(self.price_history_1s) >= self.rsi_period + 1 else None
                    ema = calculate_ema(self.price_history_1s, self.rsi_period) if len(self.price_history_1s) >= self.rsi_period else None
                    volatility = calculate_volatility(self.price_history_1s)
                    choppiness = calculate_choppiness_index(self.price_history_1s, period=900)

                # Read market data
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                if not all([call_data, put_data]):
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Handle null best_bid/best_ask (happens when deep ITM/OTM or period end)
                call_bid_data = call_data.get('best_bid')
                call_ask_data = call_data.get('best_ask')
                put_bid_data = put_data.get('best_bid')
                put_ask_data = put_data.get('best_ask')

                call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
                call_ask = call_ask_data.get('price', 0) if call_ask_data else 0
                put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
                put_ask = put_ask_data.get('price', 0) if put_ask_data else 0

                # Store last valid prices (before buffer)
                if not in_buffer_zone:
                    if call_bid > 0:
                        self.last_call_bid = call_bid
                    if put_bid > 0:
                        self.last_put_bid = put_bid
                    if call_ask > 0:
                        self.last_call_ask = call_ask
                    if put_ask > 0:
                        self.last_put_ask = put_ask
                    if btc_price > 0:
                        self.last_btc_price = btc_price

                # Check if we have needed prices for current position
                if self.position:
                    if self.position.token_type == 'CALL' and call_bid == 0:
                        logger.debug(f"⚠️  CALL bid is null, skipping trading logic")
                        time.sleep(CHECK_INTERVAL)
                        continue
                    elif self.position.token_type == 'PUT' and put_bid == 0:
                        logger.debug(f"⚠️  PUT bid is null, skipping trading logic")
                        time.sleep(CHECK_INTERVAL)
                        continue
                else:
                    # No position - need both asks
                    if call_ask == 0 or put_ask == 0:
                        logger.debug(f"⚠️  CALL or PUT ask is null, skipping trading logic")
                        time.sleep(CHECK_INTERVAL)
                        continue

                # Calculate tolerance
                tolerance = max(MIN_TOLERANCE, volatility / 8) if volatility > 0 else MIN_TOLERANCE

                # Calculate bands
                if ema is not None:
                    upper_band = ema + tolerance
                    lower_band = ema - tolerance
                else:
                    upper_band = lower_band = None

                # Display status every 10 seconds
                if int(current_time) % 10 == 0:
                    pos_str = f"{self.position.token_type}@${self.position.entry_price:.2f}" if self.position else "NONE"
                    rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                    ema_str = f"{ema:.2f}" if ema is not None else "N/A"

                    if ema is not None and upper_band is not None:
                        btc_vs_bands = "ABOVE" if btc_price > upper_band else "BELOW" if btc_price < lower_band else "INSIDE"
                        print(f"[D2 {now.strftime('%H:%M:%S')}] "
                              f"BTC:${btc_price:.2f} {btc_vs_bands} [${lower_band:.2f} EMA:${ema_str} ${upper_band:.2f}] "
                              f"Chop:{choppiness:.1f} RSI{self.rsi_period}s={rsi_str} "
                              f"Pos:{pos_str} | PNL:{self.get_daily_pnl():+.2f} | {seconds_remaining}s")
                    else:
                        samples_have = len(self.price_history_1s)
                        samples_need = self.rsi_period + 1
                        pct = (samples_have / samples_need * 100) if samples_need > 0 else 0
                        print(f"[D2 {now.strftime('%H:%M:%S')}] "
                              f"BTC:${btc_price:.2f} | Building data... {samples_have}/{samples_need} ({pct:.0f}%) "
                              f"RSI{self.rsi_period}s={rsi_str} | {seconds_remaining}s")

                # Track max profit/loss
                if self.position:
                    if self.position.token_type == 'CALL' and call_bid > 0:
                        current_pnl = call_bid - self.position.entry_price
                        self.position.max_profit = max(self.position.max_profit, current_pnl)
                        self.position.max_loss = min(self.position.max_loss, current_pnl)
                    elif self.position.token_type == 'PUT' and put_bid > 0:
                        current_pnl = put_bid - self.position.entry_price
                        self.position.max_profit = max(self.position.max_profit, current_pnl)
                        self.position.max_loss = min(self.position.max_loss, current_pnl)

                # ========== MAKER ORDER MONITORING ==========

                # Monitor pending BUY orders
                if self.pending_buy_order_id is not None:
                    # Check if order filled
                    current_balance = self.check_token_balance(self.pending_buy_token_id)

                    if current_balance >= POSITION_SIZE * 0.8:
                        # Order filled!
                        logger.info(f"\n✅ BUY ORDER FILLED!")
                        logger.info(f"📊 Balance: {current_balance:.2f} shares")

                        details = self.pending_buy_details

                        self.position = Position(
                            token_type=self.pending_buy_token_type,
                            token_id=self.pending_buy_token_id,
                            entry_price=self.pending_buy_limit_price,
                            entry_time=time.time(),
                            quantity=current_balance,
                            entry_btc_price=details['btc_price'],
                            strike_price=details['strike_price'],
                            rsi_at_entry=details['rsi'],
                            ema_at_entry=details['ema'],
                            tolerance=details['tolerance'],
                            volatility=details['volatility'],
                            choppiness=details['choppiness'],
                            rsi_period=self.rsi_period,
                            time_to_expiry_at_entry=details['seconds_remaining']
                        )

                        logger.info(f"📊 Position: {current_balance:.2f} @ ${self.pending_buy_limit_price:.4f}")

                        # Clear pending order tracking
                        self.pending_buy_order_id = None
                        self.pending_buy_order_time = None
                        self.pending_buy_token_id = None
                        self.pending_buy_token_type = None
                        self.pending_buy_limit_price = None
                        self.tracked_bid_price = None
                        self.pending_buy_details = None

                    else:
                        # Check if bid price changed - need to cancel and replace
                        current_bid = call_bid if self.pending_buy_token_type == 'CALL' else put_bid

                        if current_bid > 0 and abs(current_bid - self.tracked_bid_price) >= 0.01:
                            logger.info(f"\n⚠️  BID PRICE CHANGED: ${self.tracked_bid_price:.4f} → ${current_bid:.4f}")
                            logger.info(f"🔄 Canceling old order and will replace...")

                            self.trader.cancel_all_orders()

                            # Clear pending order tracking
                            self.pending_buy_order_id = None
                            self.pending_buy_order_time = None
                            self.pending_buy_token_id = None
                            self.pending_buy_token_type = None
                            self.pending_buy_limit_price = None
                            self.tracked_bid_price = None
                            self.pending_buy_details = None

                            # Will replace order in next iteration

                # Monitor pending SELL orders
                if self.pending_sell_order_id is not None:
                    # Check if order filled
                    current_balance = self.check_token_balance(self.pending_sell_token_id)

                    if current_balance < 0.5:
                        # Order filled!
                        logger.info(f"\n✅ SELL ORDER FILLED - Position CLOSED!")

                        details = self.pending_sell_details

                        # Calculate actual P&L
                        if self.position:
                            pnl = (self.pending_sell_limit_price - self.position.entry_price) * self.position.quantity

                            trade = {
                                'type': self.position.token_type,
                                'open_time': datetime.fromtimestamp(self.position.entry_time).isoformat(),
                                'close_time': datetime.now().isoformat(),
                                'open_btc': self.position.entry_btc_price,
                                'close_btc': details['btc_price'],
                                'open_price': self.position.entry_price,
                                'close_price': self.pending_sell_limit_price,
                                'strike_price': self.position.strike_price,
                                'rsi_at_entry': self.position.rsi_at_entry,
                                'ema_at_entry': self.position.ema_at_entry,
                                'tolerance': self.position.tolerance,
                                'volatility_at_entry': self.position.volatility,
                                'volatility_at_exit': details['volatility'],
                                'choppiness_at_entry': self.position.choppiness,
                                'choppiness_at_exit': details['choppiness'],
                                'rsi_period_at_entry': self.position.rsi_period,
                                'rsi_period_at_exit': self.rsi_period,
                                'max_profit': self.position.max_profit,
                                'max_loss': self.position.max_loss,
                                'time_to_expiry_at_entry': self.position.time_to_expiry_at_entry,
                                'time_to_expiry_at_exit': details['seconds_remaining'],
                                'pnl': pnl,
                                'close_reason': details['reason']
                            }

                            self.today_trades.append(trade)
                            self.save_trades()

                            logger.info(f"📈 P&L: ${pnl:.2f}")

                        self.position = None
                        self.last_position_close_time = time.time()

                        # Clear pending order tracking
                        self.pending_sell_order_id = None
                        self.pending_sell_order_time = None
                        self.pending_sell_token_id = None
                        self.pending_sell_limit_price = None
                        self.tracked_ask_price = None
                        self.pending_sell_details = None

                    else:
                        # Check if ask price changed - need to cancel and replace
                        current_ask = call_ask if self.position.token_type == 'CALL' else put_ask

                        if current_ask > 0 and abs(current_ask - self.tracked_ask_price) >= 0.01:
                            logger.info(f"\n⚠️  ASK PRICE CHANGED: ${self.tracked_ask_price:.4f} → ${current_ask:.4f}")
                            logger.info(f"🔄 Canceling old order and will replace...")

                            self.trader.cancel_all_orders()

                            # Clear pending order tracking
                            self.pending_sell_order_id = None
                            self.pending_sell_order_time = None
                            self.pending_sell_token_id = None
                            self.pending_sell_limit_price = None
                            self.tracked_ask_price = None
                            self.pending_sell_details = None

                            # Will replace order in next iteration

                # Skip trading if not ready or in buffer
                if (rsi is None or ema is None or seconds_into_period < START_DELAY or
                    self.strike_price is None or in_buffer_zone):
                    time.sleep(CHECK_INTERVAL)
                    continue

                # ========== TRADING LOGIC ==========
                signal = None

                # Priority 1: RSI Exits (only if we have a position and NO pending sell order)
                if self.position and self.pending_sell_order_id is None:
                    if self.position.token_type == 'CALL' and rsi > RSI_EXIT_CALL:
                        signal = 'CLOSE_CALL'
                        logger.info(f"\n[SIGNAL] RSI EXIT TRIGGERED! RSI={rsi:.1f} > {RSI_EXIT_CALL} | CALL position will close")
                    elif self.position.token_type == 'PUT' and rsi < RSI_EXIT_PUT:
                        signal = 'CLOSE_PUT'
                        logger.info(f"\n[SIGNAL] RSI EXIT TRIGGERED! RSI={rsi:.1f} < {RSI_EXIT_PUT} | PUT position will close")

                # Priority 2: Flip wrong position (only if we have position and NO pending orders)
                if self.position and self.pending_sell_order_id is None and self.pending_buy_order_id is None:
                    if self.position.token_type == 'CALL' and btc_price < lower_band and rsi > RSI_EXIT_PUT:
                        signal = 'FLIP_CALL_TO_PUT'
                    elif self.position.token_type == 'PUT' and btc_price > upper_band and rsi < RSI_EXIT_CALL:
                        signal = 'FLIP_PUT_TO_CALL'

                # Priority 3: Open new position (only if NO position and NO pending buy order, with choppiness filter)
                if not self.position and self.pending_buy_order_id is None and choppiness >= MIN_CHOPPINESS:
                    if btc_price > upper_band and rsi < RSI_EXIT_CALL:
                        signal = 'OPEN_CALL'
                    elif btc_price < lower_band and rsi > RSI_EXIT_PUT:
                        signal = 'OPEN_PUT'

                # Execute signals with confirmation
                if signal:
                    if self.last_signal_type != signal:
                        self.last_signal_time = current_time
                        self.last_signal_type = signal
                    elif current_time - self.last_signal_time >= CONFIRMATION_SECONDS:

                        if signal == 'OPEN_CALL':
                            # MAKER: use bid price for limit order
                            if MIN_BUY_PRICE <= call_bid <= MAX_BUY_PRICE and call_bid > 0:
                                if self.can_open_position():
                                    self.execute_buy('CALL', self.current_call_id, call_bid, btc_price,
                                                   self.strike_price, rsi, ema, tolerance, volatility,
                                                   choppiness, seconds_remaining,
                                                   f"BTC>${upper_band:.2f} RSI={rsi:.1f} Chop={choppiness:.1f}")
                            self.last_signal_type = None

                        elif signal == 'OPEN_PUT':
                            # MAKER: use bid price for limit order
                            if MIN_BUY_PRICE <= put_bid <= MAX_BUY_PRICE and put_bid > 0:
                                if self.can_open_position():
                                    self.execute_buy('PUT', self.current_put_id, put_bid, btc_price,
                                                   self.strike_price, rsi, ema, tolerance, volatility,
                                                   choppiness, seconds_remaining,
                                                   f"BTC<${lower_band:.2f} RSI={rsi:.1f} Chop={choppiness:.1f}")
                            self.last_signal_type = None

                        elif signal == 'CLOSE_CALL':
                            # MAKER: use ask price for limit sell order
                            if call_ask > 0:
                                self.execute_sell(self.position.token_id, call_ask, btc_price,
                                                volatility, choppiness, seconds_remaining, "RSI_EXIT")
                            self.last_signal_type = None

                        elif signal == 'CLOSE_PUT':
                            # MAKER: use ask price for limit sell order
                            if put_ask > 0:
                                self.execute_sell(self.position.token_id, put_ask, btc_price,
                                                volatility, choppiness, seconds_remaining, "RSI_EXIT")
                            self.last_signal_type = None

                        elif signal == 'FLIP_CALL_TO_PUT':
                            # Close CALL (MAKER: use ask price)
                            if call_ask > 0:
                                self.execute_sell(self.position.token_id, call_ask, btc_price,
                                                volatility, choppiness, seconds_remaining, "FLIP_TO_PUT")
                            self.last_signal_type = None

                        elif signal == 'FLIP_PUT_TO_CALL':
                            # Close PUT (MAKER: use ask price)
                            if put_ask > 0:
                                self.execute_sell(self.position.token_id, put_ask, btc_price,
                                                volatility, choppiness, seconds_remaining, "FLIP_TO_CALL")
                            self.last_signal_type = None
                else:
                    self.last_signal_type = None

                # Save state periodically
                if current_time - self.last_state_save >= STATE_SAVE_INTERVAL:
                    self.save_state()
                    self.last_state_save = current_time

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")

            # Save state
            self.save_state()

            # Cancel any pending orders
            if self.pending_buy_order_id or self.pending_sell_order_id:
                logger.info(f"🔄 Canceling pending orders...")
                self.trader.cancel_all_orders()

            # Close open position
            if self.position:
                logger.info(f"\n⚠️  Closing open {self.position.token_type} position...")
                data = read_json(CALL_FILE if self.position.token_type == 'CALL' else PUT_FILE)

                if data and data.get('best_ask'):
                    # MAKER: use ask price for sell
                    exit_ask_data = data.get('best_ask')
                    exit_price = exit_ask_data.get('price', 0) if exit_ask_data else 0
                    btc_data = read_json(BTC_FILE)
                    btc_price = btc_data.get('price', 0) if btc_data else 0

                    if exit_price > 0:
                        volatility = calculate_volatility(self.price_history_1s)
                        choppiness = calculate_choppiness_index(self.price_history_1s, period=900)
                        seconds_left = get_seconds_to_expiry()

                        self.execute_sell(self.position.token_id, exit_price, btc_price,
                                        volatility, choppiness, seconds_left, "MANUAL_STOP")

                        # Wait for order to fill
                        time.sleep(5)

            # Final save
            self.save_trades()
            logger.info(f"\n💾 Saved {len(self.today_trades)} trades")
            logger.info(f"📊 Daily PNL: {self.get_daily_pnl():+.3f}")


def main():
    """Main entry point"""
    try:
        # Load credentials
        try:
            env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh40.env'
            credentials = load_credentials_from_env(env_path)
            print(f"✅ Credentials loaded from {env_path}")
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return

        # Create and run bot
        bot = BotD2Production(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
