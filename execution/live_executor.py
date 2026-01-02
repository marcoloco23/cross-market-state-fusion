"""
Live execution adapter connecting RL agent to Polymarket.

This module bridges the paper trading RL agent with real Polymarket execution
using the Parallax project's PolymarketClient.
"""
import asyncio
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

# Add parent Parallax project to path for imports
PARALLAX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PARALLAX_PATH not in sys.path:
    sys.path.insert(0, PARALLAX_PATH)

# Load environment from Parallax .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PARALLAX_PATH, ".env"))


@dataclass
class ExecutionResult:
    """Result of a live trade execution."""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    filled_size: float = 0.0
    avg_price: float = 0.0
    timestamp: Optional[datetime] = None


class LiveExecutor:
    """
    Async adapter to execute RL agent trades on Polymarket.

    Uses the Parallax project's PolymarketClient for order execution.
    Respects KILL_SWITCH and DRY_RUN environment variables.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize the live executor.

        Args:
            dry_run: If True, simulate orders without real execution.
                    Can be overridden by DRY_RUN environment variable.
        """
        # Check environment overrides
        env_dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        env_kill_switch = os.getenv("KILL_SWITCH", "on").lower() == "on"

        self.dry_run = dry_run or env_dry_run
        self.kill_switch = env_kill_switch
        self._client = None
        self._client_initialized = False

        # Log configuration
        print(f"  [LiveExecutor] dry_run={self.dry_run}, kill_switch={self.kill_switch}")

    def _init_client(self):
        """Lazy initialization of Polymarket client."""
        if self._client_initialized:
            return

        try:
            from src.external.polymarket_client import PolymarketClient, PolymarketClientConfig

            config = PolymarketClientConfig(
                api_base=os.getenv("POLYMARKET_API_BASE", "https://clob.polymarket.com"),
                private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
                proxy_address=os.getenv("POLYMARKET_PROXY_ADDRESS"),
                signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
                api_key=os.getenv("POLYMARKET_API_KEY"),
                api_secret=os.getenv("POLYMARKET_API_SECRET"),
                api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
            )

            self._client = PolymarketClient(config)
            self._client_initialized = True
            print("  [LiveExecutor] Polymarket client initialized")

        except Exception as e:
            print(f"  [LiveExecutor] Failed to initialize client: {e}")
            self._client = None
            self._client_initialized = True

    def _create_order_intent(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        size_usd: float,
        limit_price: float,
    ):
        """
        Create an OrderIntent from RL agent trade data.

        Args:
            condition_id: Market condition ID
            token_id: Token ID for the outcome being traded
            outcome: "Up" or "Down"
            side: "buy" or "sell"
            size_usd: Trade size in USD
            limit_price: Limit price (probability 0.0-1.0)
        """
        from src.data.models import OrderIntent

        # Calculate token size from USD amount
        size_tokens = size_usd / limit_price if limit_price > 0 else 0

        # Generate unique idempotency key
        idempotency_key = f"rl-{condition_id[:16]}-{uuid.uuid4().hex[:8]}"

        return OrderIntent(
            market_id=condition_id,
            outcome=outcome,
            side=side.lower(),
            size_tokens=size_tokens,
            limit_price=limit_price,
            time_in_force="IOC",  # Immediate or Cancel for fast markets
            idempotency_key=idempotency_key,
            token_id=token_id,
        )

    async def execute_trade(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        size_usd: float,
        limit_price: float,
    ) -> ExecutionResult:
        """
        Execute a single trade on Polymarket.

        Args:
            condition_id: Market condition ID
            token_id: Token ID for the outcome
            outcome: "Up" or "Down"
            side: "buy" or "sell"
            size_usd: Trade size in USD
            limit_price: Limit price (probability)

        Returns:
            ExecutionResult with success status and order details
        """
        timestamp = datetime.now()

        # Safety check: kill switch
        if self.kill_switch:
            return ExecutionResult(
                success=False,
                error="KILL_SWITCH is ON - trading disabled",
                timestamp=timestamp,
            )

        # Validate inputs
        if size_usd <= 0:
            return ExecutionResult(
                success=False,
                error=f"Invalid size: {size_usd}",
                timestamp=timestamp,
            )

        if not 0 < limit_price < 1:
            return ExecutionResult(
                success=False,
                error=f"Invalid price: {limit_price}",
                timestamp=timestamp,
            )

        # Create order intent
        try:
            intent = self._create_order_intent(
                condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                side=side,
                size_usd=size_usd,
                limit_price=limit_price,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Failed to create order: {e}",
                timestamp=timestamp,
            )

        # Dry run: simulate success
        if self.dry_run:
            return ExecutionResult(
                success=True,
                order_id=f"DRY-{intent.idempotency_key}",
                filled_size=intent.size_tokens,
                avg_price=limit_price,
                timestamp=timestamp,
            )

        # Live execution
        try:
            self._init_client()

            if self._client is None:
                return ExecutionResult(
                    success=False,
                    error="Polymarket client not available",
                    timestamp=timestamp,
                )

            result = await self._client.execute_order_intent(intent)

            order_id = result.get("orderID") or result.get("order_id") or "unknown"

            return ExecutionResult(
                success=True,
                order_id=order_id,
                filled_size=intent.size_tokens,
                avg_price=limit_price,
                timestamp=timestamp,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                timestamp=timestamp,
            )

    async def get_balance(self) -> Optional[float]:
        """Get current account balance in USD."""
        try:
            self._init_client()
            if self._client:
                return await self._client.get_account_balance()
        except Exception as e:
            print(f"  [LiveExecutor] Failed to get balance: {e}")
        return None


class SyncLiveExecutor:
    """
    Synchronous wrapper for LiveExecutor.

    The RL agent's execute_action is synchronous, but PolymarketClient is async.
    This class provides a sync interface by managing an event loop.
    """

    def __init__(self, dry_run: bool = True):
        """
        Initialize the sync executor.

        Args:
            dry_run: If True, simulate orders without real execution.
        """
        self._executor = LiveExecutor(dry_run=dry_run)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get existing event loop or create a new one."""
        try:
            # Try to get running loop (if called from async context)
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop - create or reuse our own
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._get_or_create_loop()

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - use nest_asyncio or run in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=30)
        except RuntimeError:
            # Not in async context - run directly
            return loop.run_until_complete(coro)

    def execute_trade(
        self,
        condition_id: str,
        token_id: str,
        outcome: str,
        side: str,
        size_usd: float,
        limit_price: float,
    ) -> ExecutionResult:
        """
        Synchronously execute a trade on Polymarket.

        Args:
            condition_id: Market condition ID
            token_id: Token ID for the outcome
            outcome: "Up" or "Down"
            side: "buy" or "sell"
            size_usd: Trade size in USD
            limit_price: Limit price (probability)

        Returns:
            ExecutionResult with success status and order details
        """
        return self._run_async(
            self._executor.execute_trade(
                condition_id=condition_id,
                token_id=token_id,
                outcome=outcome,
                side=side,
                size_usd=size_usd,
                limit_price=limit_price,
            )
        )

    def get_balance(self) -> Optional[float]:
        """Get current account balance in USD."""
        return self._run_async(self._executor.get_balance())

    @property
    def dry_run(self) -> bool:
        """Check if executor is in dry run mode."""
        return self._executor.dry_run

    @property
    def kill_switch(self) -> bool:
        """Check if kill switch is enabled."""
        return self._executor.kill_switch
