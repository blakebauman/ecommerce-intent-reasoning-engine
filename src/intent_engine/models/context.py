"""Context models for order and customer data enrichment."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class CustomerTier(str, Enum):
    """Customer tier for policy decisions."""

    VIP = "vip"  # High lifetime value, priority support
    PREMIUM = "premium"  # Paid membership or high value
    STANDARD = "standard"  # Regular customer
    NEW = "new"  # First-time customer
    AT_RISK = "at_risk"  # Multiple complaints, might churn
    FLAGGED = "flagged"  # Fraud risk or abuse pattern


class WarrantyStatus(str, Enum):
    """Warranty status for products."""

    ACTIVE = "active"
    EXPIRED = "expired"
    EXTENDED = "extended"
    NOT_APPLICABLE = "not_applicable"


class ReturnEligibility(str, Enum):
    """Return eligibility status."""

    ELIGIBLE = "eligible"
    EXPIRED = "expired"
    FINAL_SALE = "final_sale"
    EXCHANGE_ONLY = "exchange_only"
    REQUIRES_APPROVAL = "requires_approval"


class CustomerProfile(BaseModel):
    """Customer profile for context enrichment."""

    customer_id: str
    email: str
    name: str | None = None

    # Tier and value
    tier: CustomerTier = CustomerTier.STANDARD
    lifetime_value: float = Field(default=0.0, ge=0)
    total_orders: int = Field(default=0, ge=0)
    first_order_date: datetime | None = None

    # Engagement metrics
    support_tickets_30d: int = Field(default=0, ge=0)
    complaints_90d: int = Field(default=0, ge=0)
    returns_90d: int = Field(default=0, ge=0)

    # Preferences
    preferred_contact: str | None = None  # email, sms, phone
    language: str = "en"
    timezone: str | None = None

    # Flags
    is_vip: bool = False
    is_at_risk: bool = False
    requires_escalation: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "cust-12345",
                    "email": "john@example.com",
                    "name": "John Doe",
                    "tier": "premium",
                    "lifetime_value": 2500.00,
                    "total_orders": 15,
                }
            ]
        }
    }


class ProductContext(BaseModel):
    """Product context for policy decisions."""

    product_id: str
    sku: str | None = None
    name: str
    category: str | None = None
    price: float = Field(ge=0)
    currency: str = "USD"

    # Return policy
    is_returnable: bool = True
    return_window_days: int = 30
    is_final_sale: bool = False
    exchange_only: bool = False

    # Warranty
    warranty_status: WarrantyStatus = WarrantyStatus.NOT_APPLICABLE
    warranty_expires: datetime | None = None

    # Stock
    is_in_stock: bool = True
    restock_date: datetime | None = None


class OrderContext(BaseModel):
    """
    Enriched order context for intent resolution.

    Combines order data from platform (Shopify) with
    computed policy data.
    """

    # Order identification
    order_id: str
    order_number: str
    platform: str = "shopify"

    # Status
    status: str
    fulfillment_status: str
    is_cancelled: bool = False

    # Customer
    customer_id: str | None = None
    customer_email: str
    customer_name: str | None = None

    # Order details
    items: list[ProductContext] = Field(default_factory=list)
    subtotal: float = Field(ge=0)
    shipping_cost: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    total: float = Field(ge=0)
    currency: str = "USD"

    # Timestamps
    created_at: datetime | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None

    # Shipping
    shipping_address: dict[str, str] | None = None  # Simplified for context
    tracking_number: str | None = None
    carrier: str | None = None
    tracking_url: str | None = None
    estimated_delivery: datetime | None = None

    # Return/refund policy data
    return_eligibility: ReturnEligibility = ReturnEligibility.ELIGIBLE
    return_window_ends: datetime | None = None
    days_until_return_expires: int | None = None
    is_within_return_window: bool = True

    # Refund info
    refund_amount: float | None = None
    is_fully_refunded: bool = False
    is_partially_refunded: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "order_id": "12345",
                    "order_number": "#1234",
                    "status": "delivered",
                    "fulfillment_status": "fulfilled",
                    "customer_email": "john@example.com",
                    "total": 129.99,
                    "is_within_return_window": True,
                    "days_until_return_expires": 15,
                }
            ]
        }
    }


class EnrichedContext(BaseModel):
    """
    Complete enriched context for intent resolution.

    Contains customer profile, order context, and policy decisions.
    """

    # Customer data
    customer: CustomerProfile | None = None

    # Order data (for the relevant order, if any)
    order: OrderContext | None = None

    # Policy decisions (computed by policy engine)
    auto_approve_return: bool = False
    auto_approve_refund: bool = False
    escalation_required: bool = False
    escalation_reason: str | None = None

    # Additional orders in recent history
    recent_orders: list[OrderContext] = Field(default_factory=list)

    # Support history context
    open_tickets: int = 0
    recent_complaints: int = 0

    # Enrichment metadata
    enriched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_sources: list[str] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer": {
                        "customer_id": "cust-123",
                        "email": "john@example.com",
                        "tier": "premium",
                    },
                    "order": {"order_id": "12345", "order_number": "#1234", "status": "delivered"},
                    "auto_approve_return": True,
                }
            ]
        }
    }


class PolicyContext(BaseModel):
    """Policy configuration context for a tenant."""

    tenant_id: str

    # Return policy
    default_return_window_days: int = 30
    extended_return_window_vip: int = 60
    final_sale_categories: list[str] = Field(default_factory=list)

    # Auto-approval thresholds
    auto_approve_return_max_amount: float = 100.0
    auto_approve_refund_max_amount: float = 50.0
    auto_approve_vip_max_amount: float = 500.0

    # Escalation rules
    escalation_complaint_threshold: int = 3  # 3+ complaints → escalate
    escalation_amount_threshold: float = 500.0  # High-value orders
    escalation_required_categories: list[str] = Field(default_factory=list)

    # Contact preferences
    priority_response_sla_minutes: int = 60
    standard_response_sla_minutes: int = 240

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenant_id": "merchant-1",
                    "default_return_window_days": 30,
                    "auto_approve_return_max_amount": 150.0,
                }
            ]
        }
    }
