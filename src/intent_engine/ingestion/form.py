"""Form channel adapter with field concatenation and metadata extraction."""

import re
import uuid
from datetime import datetime
from typing import Any

from intent_engine.ingestion.base import ChannelAdapter
from intent_engine.models.request import InputChannel, IntentRequest


class FormAdapter(ChannelAdapter):
    """
    Adapter for web form/contact form channel input.

    Handles:
    - Field concatenation to raw_text
    - Metadata extraction (field names → entity hints)
    - Common form field mapping (name, email, order_id, etc.)
    - Subject/topic field handling

    Expected input format:
    {
        "tenant_id": "merchant-1",
        "form_id": "contact-form-1",  # optional
        "fields": {
            "name": "John Doe",
            "email": "john@example.com",
            "order_number": "ORD-12345",
            "subject": "Return request",
            "message": "I want to return my order because it doesn't fit.",
            "issue_type": "return"  # optional dropdown value
        },
        "page_url": "https://store.com/contact",  # optional
        "timestamp": "2024-02-09T10:30:00Z",  # optional
        "customer_id": "cust-456"  # optional
    }
    """

    # Common field name mappings to entity types
    FIELD_MAPPINGS = {
        # Order ID fields
        "order_number": "order_id",
        "order_id": "order_id",
        "ordernumber": "order_id",
        "orderid": "order_id",
        "order": "order_id",
        "order_ref": "order_id",
        "reference": "order_id",
        "confirmation_number": "order_id",
        # Customer name fields
        "name": "customer_name",
        "full_name": "customer_name",
        "fullname": "customer_name",
        "customer_name": "customer_name",
        # Email fields
        "email": "customer_email",
        "email_address": "customer_email",
        "customer_email": "customer_email",
        # Phone fields
        "phone": "customer_phone",
        "phone_number": "customer_phone",
        "telephone": "customer_phone",
        "mobile": "customer_phone",
        # Message/body fields
        "message": "message",
        "body": "message",
        "description": "message",
        "details": "message",
        "inquiry": "message",
        "question": "message",
        "comment": "message",
        "comments": "message",
        "feedback": "message",
        # Subject/topic fields
        "subject": "subject",
        "topic": "subject",
        "issue": "subject",
        "reason": "subject",
        "issue_type": "issue_type",
        "category": "issue_type",
        "request_type": "issue_type",
        # Product fields
        "product": "product_name",
        "product_name": "product_name",
        "item": "product_name",
        "sku": "product_sku",
        "product_sku": "product_sku",
    }

    # Fields that contain the main message content
    MESSAGE_FIELDS = {"message", "body", "description", "details", "inquiry", "question", "comment", "comments", "feedback"}

    # Fields that provide context/subject
    SUBJECT_FIELDS = {"subject", "topic", "issue", "reason"}

    # Fields to exclude from concatenation (metadata only)
    METADATA_ONLY_FIELDS = {
        "email", "email_address", "customer_email",
        "phone", "phone_number", "telephone", "mobile",
        "name", "full_name", "fullname", "customer_name",
        "customer_id", "session_id",
    }

    @property
    def channel_name(self) -> str:
        return "form"

    def validate(self, raw_input: dict[str, Any]) -> bool:
        """Validate form input structure."""
        # Must have fields dict and tenant_id
        if "fields" not in raw_input or not isinstance(raw_input["fields"], dict):
            return False

        if "tenant_id" not in raw_input:
            return False

        # Must have at least some content
        fields = raw_input["fields"]
        has_content = any(
            str(v).strip()
            for k, v in fields.items()
            if k.lower() in self.MESSAGE_FIELDS or k.lower() in self.SUBJECT_FIELDS
        )

        return has_content

    async def normalize(self, raw_input: dict[str, Any]) -> IntentRequest:
        """
        Normalize form input to IntentRequest.

        Args:
            raw_input: Raw form submission data.

        Returns:
            Normalized IntentRequest.
        """
        if not self.validate(raw_input):
            raise ValueError("Invalid form input: missing required fields or content")

        fields = raw_input["fields"]
        tenant_id = raw_input["tenant_id"]

        # Extract and classify fields
        classified = self._classify_fields(fields)

        # Build raw_text from relevant fields
        raw_text = self._build_raw_text(classified, fields)

        # Extract order IDs from all text content
        order_ids = self.extract_order_ids(raw_text)

        # Also check specific order fields
        if order_id := classified.get("order_id"):
            if order_id not in order_ids:
                order_ids.append(order_id)

        # Generate request ID
        request_id = raw_input.get("request_id", str(uuid.uuid4()))

        # Parse timestamp
        timestamp = datetime.utcnow()
        if ts := raw_input.get("timestamp"):
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif isinstance(ts, datetime):
                timestamp = ts

        # Build metadata with entity hints
        raw_metadata: dict[str, Any] = {
            "form_id": raw_input.get("form_id"),
            "page_url": raw_input.get("page_url"),
            "entity_hints": self._build_entity_hints(classified),
            "original_fields": list(fields.keys()),
            "issue_type": classified.get("issue_type"),
        }

        # Customer info from form
        customer_email = classified.get("customer_email")

        return IntentRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            channel=InputChannel.FORM,
            timestamp=timestamp,
            raw_text=raw_text,
            raw_metadata=raw_metadata,
            customer_id=raw_input.get("customer_id"),
            customer_tier=raw_input.get("customer_tier"),
            order_ids=order_ids,
            page_context=raw_input.get("page_url"),
        )

    def _classify_fields(self, fields: dict[str, Any]) -> dict[str, str]:
        """Classify form fields by their semantic type."""
        classified: dict[str, str] = {}

        for field_name, value in fields.items():
            if not value:
                continue

            # Normalize field name for matching
            normalized = field_name.lower().replace("-", "_").replace(" ", "_")

            # Look up in mapping
            if normalized in self.FIELD_MAPPINGS:
                field_type = self.FIELD_MAPPINGS[normalized]
                classified[field_type] = str(value).strip()
            else:
                # Store unmapped fields as-is
                classified[f"raw_{normalized}"] = str(value).strip()

        return classified

    def _build_raw_text(self, classified: dict[str, str], original_fields: dict[str, Any]) -> str:
        """Build the raw_text from classified fields."""
        parts: list[str] = []

        # 1. Add issue type if present (gives context)
        if issue_type := classified.get("issue_type"):
            parts.append(f"Issue Type: {issue_type}")

        # 2. Add subject if present
        if subject := classified.get("subject"):
            parts.append(f"Subject: {subject}")

        # 3. Add main message content
        if message := classified.get("message"):
            parts.append(message)

        # 4. If no message, concatenate any unclassified text fields
        if not classified.get("message"):
            for field_name, value in original_fields.items():
                if not value:
                    continue

                normalized = field_name.lower().replace("-", "_").replace(" ", "_")

                # Skip metadata-only fields
                if normalized in self.METADATA_ONLY_FIELDS:
                    continue

                # Skip already processed fields
                if normalized in self.SUBJECT_FIELDS:
                    continue

                # Skip known non-text fields
                if normalized in {"issue_type", "category", "request_type"}:
                    continue

                # Add the field value
                value_str = str(value).strip()
                if value_str:
                    if len(parts) > 0:
                        parts.append(f"{field_name}: {value_str}")
                    else:
                        parts.append(value_str)

        # Combine parts
        raw_text = "\n\n".join(parts)

        # If still empty, try to build from any available text
        if not raw_text.strip():
            all_text = " ".join(str(v) for v in original_fields.values() if v)
            raw_text = all_text.strip()

        return raw_text

    def _build_entity_hints(self, classified: dict[str, str]) -> dict[str, str]:
        """Build entity hints from classified fields."""
        hints: dict[str, str] = {}

        # Map classified fields to entity hints
        hint_mappings = {
            "order_id": "order_id",
            "customer_name": "person_name",
            "customer_email": "email",
            "customer_phone": "phone",
            "product_name": "product_name",
            "product_sku": "product_sku",
        }

        for classified_type, entity_type in hint_mappings.items():
            if value := classified.get(classified_type):
                hints[entity_type] = value

        return hints

    def extract_order_ids(self, text: str) -> list[str]:
        """Extract order IDs from text with additional form-specific patterns."""
        # Use base class patterns
        order_ids = super().extract_order_ids(text)

        # Add form-specific patterns (confirmation codes, etc.)
        additional_patterns = [
            r"\b(\d{4,10})\b",  # Plain numbers that could be order IDs
        ]

        # Only add plain numbers if they look like order IDs (4-10 digits standalone)
        # This is more aggressive but form contexts are usually more structured

        return list(set(order_ids))
