"""
Standalone logger demo test (not related to TBML project logic).

Run:
    python utilities/logger_demo_test.py
"""

from logger import setup_logger, log_event, log_exception


def process_order(order_id: str, quantity: int) -> None:
    logger = setup_logger("demo.order-flow")

    log_event(
        logger,
        headline="Order processing initiated",
        status="INFO",
        message="Starting order pipeline",
        order_id=order_id,
        quantity=quantity,
    )

    if quantity <= 0:
        log_event(
            logger,
            headline="Invalid order quantity",
            status="WARNING",
            message="Quantity should be greater than zero",
            order_id=order_id,
            quantity=quantity,
        )
        return

    log_event(
        logger,
        headline="Payment step",
        status="INFO",
        message="Payment gateway called",
        order_id=order_id,
    )

    log_event(
        logger,
        headline="Inventory reserved",
        status="SUCCESS",
        message="Stock allocated successfully",
        order_id=order_id,
    )

    # Demonstrate exception logging for a simulated shipping failure.
    try:
        raise RuntimeError("Shipping service timeout")
    except RuntimeError as error:
        log_exception(
            logger,
            headline="Shipping allocation failed",
            error=error,
            order_id=order_id,
            retryable=True,
        )


def main() -> None:
    process_order("ORD-1001", 3)
    process_order("ORD-1002", 0)


if __name__ == "__main__":
    main()
