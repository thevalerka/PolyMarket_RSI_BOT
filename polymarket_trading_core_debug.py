#!/usr/bin/env python3
"""
Polymarket Trading Core Functions

All essential functions needed to buy and sell on Polymarket.
Copy these functions to build any trading bot.
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

class PolymarketTrader:
    """Core Polymarket trading functions"""

    def __init__(self,
                 clob_api_url: str,
                 private_key: str,
                 api_key: str,
                 api_secret: str,
                 api_passphrase: str,
                 chain_id: int = 137):
        """
        Initialize Polymarket trader

        Args:
            clob_api_url: Polymarket CLOB API URL
            private_key: Your wallet private key
            api_key: CLOB API key
            api_secret: CLOB API secret
            api_passphrase: CLOB API passphrase
            chain_id: Polygon chain ID (137)
        """
        self.client = self._setup_client(clob_api_url, private_key, api_key, api_secret, api_passphrase, chain_id)
        print("✅ Polymarket trader initialized")

    def _setup_client(self, clob_api_url: str, private_key: str, api_key: str,
                     api_secret: str, api_passphrase: str, chain_id: int) -> ClobClient:
        """Setup CLOB client with credentials"""
        try:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase
            )

            client = ClobClient(
                clob_api_url,
                key=private_key,
                chain_id=chain_id,
                creds=creds
            )

            return client

        except Exception as e:
            raise Exception(f"Failed to setup CLOB client: {e}")

    def get_token_balance(self, token_id: str) -> Tuple[int, float]:
        """
        Get balance for a specific token

        Args:
            token_id: The token ID to check balance for

        Returns:
            Tuple of (raw_balance, token_balance)
        """
        try:
            print(f"      🔍 DEBUG get_token_balance: Requesting balance for token_id={token_id[:16]}...")

            response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id
                )
            )

            print(f"      🔍 DEBUG get_token_balance: API response type={type(response)}")
            print(f"      🔍 DEBUG get_token_balance: API response={response}")

            balance_raw = int(response.get('balance', 0))
            balance_tokens = balance_raw / 10**6  # Convert from raw units

            print(f"      🔍 DEBUG get_token_balance: balance_raw={balance_raw}, balance_tokens={balance_tokens:.6f}")

            return balance_raw, balance_tokens

        except Exception as e:
            print(f"❌ Error getting balance for token {token_id[:8]}...: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0.0

    def place_buy_order(self, token_id: str, price: float, quantity: float) -> Optional[str]:
        """
        Place a BUY order

        Args:
            token_id: Token ID to buy
            price: Price per token (in USD)
            quantity: Number of tokens to buy

        Returns:
            Order ID if successful, None if failed
        """
        try:
            order_args = OrderArgs(
                price=price,
                size=quantity,
                side=BUY,
                token_id=token_id
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)

            print(f"   🔍 DEBUG place_buy_order response: {response}")

            # Check for success (API returns 'orderID' uppercase, not 'orderId')
            order_id = response.get('orderID') or response.get('orderId')
            success = response.get('success', False)

            if order_id and success:
                print(f"✅ BUY order placed: {quantity} tokens @ ${price:.4f} = ${quantity * price:.2f}")
                print(f"   Order ID: {order_id}")
                return order_id
            elif order_id:
                # Has order ID but success not explicitly True
                print(f"✅ BUY order placed (implicit): {quantity} tokens @ ${price:.4f} = ${quantity * price:.2f}")
                print(f"   Order ID: {order_id}")
                return order_id
            else:
                print(f"❌ BUY order failed: {response}")
                return None

        except Exception as e:
            print(f"❌ Error placing BUY order: {e}")
            return None

    def place_sell_order(self, token_id: str, price: float, quantity: float) -> Optional[str]:
        """
        Place a SELL order

        Args:
            token_id: Token ID to sell
            price: Price per token (in USD)
            quantity: Number of tokens to sell

        Returns:
            Order ID if successful, None if failed
        """
        try:
            # Check if we have enough balance to sell
            balance_raw, balance_tokens = self.get_token_balance(token_id)

            if balance_tokens < quantity:
                print(f"❌ Insufficient balance: {balance_tokens:.2f} < {quantity:.2f}")
                return None

            order_args = OrderArgs(
                price=price,
                size=quantity,
                side=SELL,
                
                token_id=token_id
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)

            print(f"   🔍 DEBUG place_sell_order response: {response}")

            # Check for success (API returns 'orderID' uppercase, not 'orderId')
            order_id = response.get('orderID') or response.get('orderId')
            success = response.get('success', False)

            if order_id and success:
                print(f"✅ SELL order placed: {quantity} tokens @ ${price:.4f} = ${quantity * price:.2f}")
                print(f"   Order ID: {order_id}")
                return order_id
            elif order_id:
                # Has order ID but success not explicitly True
                print(f"✅ SELL order placed (implicit): {quantity} tokens @ ${price:.4f} = ${quantity * price:.2f}")
                print(f"   Order ID: {order_id}")
                return order_id
            else:
                print(f"❌ SELL order failed: {response}")
                return None

        except Exception as e:
            print(f"❌ Error placing SELL order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False if failed
        """
        try:
            response = self.client.cancel_order(order_id)
            print(f"✅ Order cancelled: {order_id}")
            return True

        except Exception as e:
            print(f"❌ Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders

        Returns:
            Number of orders cancelled
        """
        try:
            response = self.client.cancel_all()
            cancelled_count = len(response.get('cancelled', []))
            print(f"✅ Cancelled {cancelled_count} orders")
            return cancelled_count

        except Exception as e:
            print(f"❌ Error cancelling all orders: {e}")
            return 0

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get status of a specific order

        Args:
            order_id: Order ID to check

        Returns:
            Order status dict or None
        """
        try:
            response = self.client.get_order(order_id)
            return response

        except Exception as e:
            print(f"❌ Error getting order status for {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """
        Get all open orders

        Returns:
            List of open orders
        """
        try:
            response = self.client.get_orders()

            print(f"      🔍 DEBUG get_open_orders: response type={type(response)}")

            # Response might be a list directly, or a dict with 'orders' key
            if isinstance(response, list):
                print(f"      🔍 DEBUG get_open_orders: Got list directly with {len(response)} orders")
                return response
            elif isinstance(response, dict):
                orders = response.get('orders', [])
                print(f"      🔍 DEBUG get_open_orders: Got dict with {len(orders)} orders")
                return orders
            else:
                print(f"      ⚠️  Unexpected response type: {type(response)}")
                return []

        except Exception as e:
            print(f"❌ Error getting open orders: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_market_price(self, token_id: str) -> Optional[Dict]:
        """
        Get current market price for a token

        Args:
            token_id: Token ID to get price for

        Returns:
            Dict with bid, ask, and mid prices
        """
        try:
            response = self.client.get_price(token_id)

            bid = float(response.get('bid', 0))
            ask = float(response.get('ask', 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': ask - bid if ask > bid else 0
            }

        except Exception as e:
            print(f"❌ Error getting market price for {token_id[:8]}...: {e}")
            return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_credentials_from_env(env_file_path: str = None) -> Dict[str, str]:
    """
    Load Polymarket credentials from environment variables

    Args:
        env_file_path: Optional path to .env file

    Returns:
        Dict with credential keys
    """
    if env_file_path:
        from dotenv import load_dotenv
        load_dotenv(env_file_path)

    credentials = {
        'clob_api_url': os.getenv('CLOB_API_URL'),
        'private_key': os.getenv('PK'),
        'api_key': os.getenv('CLOB_API_KEY'),
        'api_secret': os.getenv('CLOB_SECRET'),
        'api_passphrase': os.getenv('CLOB_PASS_PHRASE')
    }

    # Validate all credentials are present
    missing = [key for key, value in credentials.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    return credentials

def read_market_data_from_json(file_path: str) -> Optional[Dict]:
    """
    Read market data from JSON file (like CALL.json, PUT.json)

    Args:
        file_path: Path to JSON file

    Returns:
        Market data dict or None
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract key information
        market_data = {
            'asset_id': data.get('asset_id'),
            'asset_name': data.get('asset_name'),
            'best_bid': data.get('best_bid', {}),
            'best_ask': data.get('best_ask', {}),
            'spread': data.get('spread', 0),
            'timestamp': data.get('timestamp'),
            'complete_book': data.get('complete_book', {})
        }

        # Calculate mid price
        bid_price = market_data['best_bid'].get('price', 0)
        ask_price = market_data['best_ask'].get('price', 0)

        if bid_price > 0 and ask_price > 0:
            market_data['mid_price'] = (bid_price + ask_price) / 2
            market_data['spread_pct'] = (ask_price - bid_price) / market_data['mid_price']
        else:
            market_data['mid_price'] = 0
            market_data['spread_pct'] = 0

        return market_data

    except Exception as e:
        print(f"❌ Error reading market data from {file_path}: {e}")
        return None

def find_large_orders(market_data: Dict, min_size: float = 1000) -> Dict[str, List[Dict]]:
    """
    Find large orders in the order book

    Args:
        market_data: Market data from read_market_data_from_json()
        min_size: Minimum order size to consider "large"

    Returns:
        Dict with 'large_bids' and 'large_asks' lists
    """
    complete_book = market_data.get('complete_book', {})
    bids = complete_book.get('bids', [])
    asks = complete_book.get('asks', [])

    large_bids = []
    large_asks = []

    # Find large bids
    for bid in bids:
        if isinstance(bid, dict):
            size = float(bid.get('size', 0))
            if size >= min_size:
                large_bids.append({
                    'price': float(bid.get('price', 0)),
                    'size': size
                })

    # Find large asks
    for ask in asks:
        if isinstance(ask, dict):
            size = float(ask.get('size', 0))
            if size >= min_size:
                large_asks.append({
                    'price': float(ask.get('price', 0)),
                    'size': size
                })

    return {
        'large_bids': large_bids,
        'large_asks': large_asks
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the Polymarket trading functions"""

    # 1. Load credentials
    try:
        creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')
        print("✅ Credentials loaded")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return

    # 2. Initialize trader
    try:
        trader = PolymarketTrader(
            clob_api_url=creds['clob_api_url'],
            private_key=creds['private_key'],
            api_key=creds['api_key'],
            api_secret=creds['api_secret'],
            api_passphrase=creds['api_passphrase']
        )
        print("✅ Trader initialized")
    except Exception as e:
        print(f"❌ Error initializing trader: {e}")
        return

    # 3. Read market data
    call_data = read_market_data_from_json('/path/to/CALL.json')
    if call_data:
        token_id = call_data['asset_id']
        print(f"✅ Token ID: {token_id}")

        # 4. Check balance
        balance_raw, balance_tokens = trader.get_token_balance(token_id)
        print(f"💰 Balance: {balance_tokens} tokens")

        # 5. Get market price
        market_price = trader.get_market_price(token_id)
        if market_price:
            print(f"📊 Market: ${market_price['bid']:.4f} / ${market_price['ask']:.4f}")

            # 6. Place a buy order
            buy_price = market_price['bid'] + 0.01  # Slightly above best bid
            buy_quantity = 10.0

            order_id = trader.place_buy_order(token_id, buy_price, buy_quantity)

            if order_id:
                # 7. Check order status
                time.sleep(1)
                order_status = trader.get_order_status(order_id)
                print(f"📋 Order status: {order_status.get('status', 'unknown')}")

                # 8. Cancel order after 5 seconds
                time.sleep(5)
                trader.cancel_order(order_id)

if __name__ == "__main__":
    print("🏪 POLYMARKET TRADING CORE FUNCTIONS")
    print("=" * 50)
    print("This file contains all essential functions for Polymarket trading.")
    print("Copy these functions to build any trading bot.")
    print()
    print("Key functions:")
    print("- place_buy_order(token_id, price, quantity)")
    print("- place_sell_order(token_id, price, quantity)")
    print("- cancel_order(order_id)")
    print("- get_token_balance(token_id)")
    print("- get_market_price(token_id)")
    print("- read_market_data_from_json(file_path)")
    print()
    print("Run example_usage() to see how to use these functions.")
