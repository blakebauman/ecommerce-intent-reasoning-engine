"""Tests for form channel adapter."""

import pytest
from datetime import datetime

from intent_engine.ingestion.form import FormAdapter
from intent_engine.models.request import InputChannel


@pytest.fixture
def adapter() -> FormAdapter:
    """Create a form adapter instance."""
    return FormAdapter()


class TestFormAdapterValidation:
    """Tests for form input validation."""

    def test_valid_form_with_message(self, adapter: FormAdapter) -> None:
        """Test validation of valid form with message field."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "message": "I want to return my order",
                "email": "customer@example.com",
            },
        }
        assert adapter.validate(raw_input) is True

    def test_valid_form_with_subject(self, adapter: FormAdapter) -> None:
        """Test validation of valid form with subject field."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "subject": "Return request",
                "name": "John Doe",
            },
        }
        assert adapter.validate(raw_input) is True

    def test_invalid_missing_fields(self, adapter: FormAdapter) -> None:
        """Test validation fails without fields dict."""
        raw_input = {
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is False

    def test_invalid_missing_tenant(self, adapter: FormAdapter) -> None:
        """Test validation fails without tenant_id."""
        raw_input = {
            "fields": {
                "message": "Help",
            },
        }
        assert adapter.validate(raw_input) is False

    def test_invalid_empty_content(self, adapter: FormAdapter) -> None:
        """Test validation fails with only metadata fields."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "email": "test@example.com",
                "name": "John",
            },
        }
        assert adapter.validate(raw_input) is False

    def test_valid_with_description_field(self, adapter: FormAdapter) -> None:
        """Test validation with description field (alternative to message)."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "description": "Problem with my order",
            },
        }
        assert adapter.validate(raw_input) is True


class TestFormAdapterNormalization:
    """Tests for form normalization."""

    @pytest.mark.asyncio
    async def test_normalize_simple_form(self, adapter: FormAdapter) -> None:
        """Test normalization of simple form."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "message": "I want to return my order #ORD-12345",
                "email": "customer@example.com",
            },
        }

        request = await adapter.normalize(raw_input)

        assert request.channel == InputChannel.FORM
        assert request.tenant_id == "merchant-1"
        assert "return my order" in request.raw_text
        assert "ORD-12345" in request.order_ids

    @pytest.mark.asyncio
    async def test_normalize_form_with_all_fields(self, adapter: FormAdapter) -> None:
        """Test normalization with all common fields."""
        raw_input = {
            "tenant_id": "merchant-1",
            "form_id": "contact-form",
            "fields": {
                "name": "John Doe",
                "email": "john@example.com",
                "order_number": "ORD-99999",
                "subject": "Return request",
                "issue_type": "return",
                "message": "I need to return the shoes I ordered.",
            },
            "page_url": "https://store.com/contact",
        }

        request = await adapter.normalize(raw_input)

        assert "Return request" in request.raw_text
        assert "return the shoes" in request.raw_text
        assert "Issue Type: return" in request.raw_text
        assert request.raw_metadata["form_id"] == "contact-form"
        assert request.raw_metadata["page_url"] == "https://store.com/contact"
        assert "ORD-99999" in request.order_ids

    @pytest.mark.asyncio
    async def test_normalize_preserves_entity_hints(self, adapter: FormAdapter) -> None:
        """Test that entity hints are captured."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "order_number": "ORD-123",
                "email": "test@example.com",
                "phone": "555-1234",
                "product": "Blue Widget",
                "message": "Question about product",
            },
        }

        request = await adapter.normalize(raw_input)
        hints = request.raw_metadata["entity_hints"]

        assert hints["order_id"] == "ORD-123"
        assert hints["email"] == "test@example.com"
        assert hints["phone"] == "555-1234"
        assert hints["product_name"] == "Blue Widget"

    @pytest.mark.asyncio
    async def test_normalize_with_timestamp(self, adapter: FormAdapter) -> None:
        """Test timestamp parsing."""
        raw_input = {
            "tenant_id": "merchant-1",
            "fields": {
                "message": "Test",
            },
            "timestamp": "2024-03-15T14:30:00Z",
        }

        request = await adapter.normalize(raw_input)

        assert request.timestamp.year == 2024
        assert request.timestamp.month == 3
        assert request.timestamp.day == 15

    @pytest.mark.asyncio
    async def test_normalize_invalid_raises(self, adapter: FormAdapter) -> None:
        """Test normalization raises for invalid input."""
        raw_input = {"tenant_id": "test", "fields": {}}

        with pytest.raises(ValueError, match="Invalid form input"):
            await adapter.normalize(raw_input)


class TestFieldClassification:
    """Tests for form field classification."""

    def test_classify_order_id_variants(self, adapter: FormAdapter) -> None:
        """Test classification of order ID field variants."""
        fields = {
            "order_number": "123",
            "orderId": "456",
            "order_ref": "789",
        }
        classified = adapter._classify_fields(fields)

        # All should map to order_id
        assert classified.get("order_id") is not None

    def test_classify_email_variants(self, adapter: FormAdapter) -> None:
        """Test classification of email field variants."""
        fields = {
            "email_address": "test@example.com",
        }
        classified = adapter._classify_fields(fields)
        assert classified["customer_email"] == "test@example.com"

    def test_classify_message_variants(self, adapter: FormAdapter) -> None:
        """Test classification of message field variants."""
        for field_name in ["message", "body", "description", "details", "inquiry"]:
            fields = {field_name: "Test content"}
            classified = adapter._classify_fields(fields)
            assert classified.get("message") == "Test content"

    def test_classify_unknown_field(self, adapter: FormAdapter) -> None:
        """Test unknown fields are prefixed with raw_."""
        fields = {"custom_field": "custom_value"}
        classified = adapter._classify_fields(fields)
        assert classified["raw_custom_field"] == "custom_value"

    def test_classify_normalizes_field_names(self, adapter: FormAdapter) -> None:
        """Test field name normalization."""
        fields = {
            "Order-Number": "123",
            "email address": "test@test.com",
        }
        classified = adapter._classify_fields(fields)
        assert classified.get("order_id") == "123"
        assert classified.get("customer_email") == "test@test.com"

    def test_classify_empty_values_skipped(self, adapter: FormAdapter) -> None:
        """Test empty values are not classified."""
        fields = {
            "message": "Real content",
            "subject": "",
        }
        classified = adapter._classify_fields(fields)
        assert "subject" not in classified


class TestRawTextBuilding:
    """Tests for raw_text construction."""

    def test_includes_issue_type(self, adapter: FormAdapter) -> None:
        """Test issue_type is included first."""
        classified = {"issue_type": "return", "message": "I want to return"}
        result = adapter._build_raw_text(classified, {})
        assert result.startswith("Issue Type: return")

    def test_includes_subject(self, adapter: FormAdapter) -> None:
        """Test subject is included."""
        classified = {"subject": "Order Help", "message": "Need assistance"}
        result = adapter._build_raw_text(classified, {})
        assert "Subject: Order Help" in result
        assert "Need assistance" in result

    def test_message_only(self, adapter: FormAdapter) -> None:
        """Test with only message field."""
        classified = {"message": "Simple message"}
        result = adapter._build_raw_text(classified, {})
        assert result == "Simple message"

    def test_fallback_to_any_text(self, adapter: FormAdapter) -> None:
        """Test fallback when no classified message."""
        classified = {}
        original = {"random_field": "Some text here"}
        result = adapter._build_raw_text(classified, original)
        assert "Some text here" in result

    def test_skips_metadata_fields_when_message_present(self, adapter: FormAdapter) -> None:
        """Test metadata-only fields are not in raw_text when message exists."""
        classified = {"message": "My order is late"}
        original = {"email": "test@test.com", "phone": "555-1234", "message": "My order is late"}
        result = adapter._build_raw_text(classified, original)
        # Email and phone should not be included when we have a message
        assert "test@test.com" not in result
        assert "555-1234" not in result
        assert "My order is late" in result


class TestEntityHints:
    """Tests for entity hint extraction."""

    def test_build_entity_hints(self, adapter: FormAdapter) -> None:
        """Test entity hint building."""
        classified = {
            "order_id": "ORD-123",
            "customer_name": "John Doe",
            "customer_email": "john@test.com",
            "customer_phone": "555-9999",
            "product_name": "Widget",
            "product_sku": "SKU-456",
        }
        hints = adapter._build_entity_hints(classified)

        assert hints["order_id"] == "ORD-123"
        assert hints["person_name"] == "John Doe"
        assert hints["email"] == "john@test.com"
        assert hints["phone"] == "555-9999"
        assert hints["product_name"] == "Widget"
        assert hints["product_sku"] == "SKU-456"

    def test_partial_hints(self, adapter: FormAdapter) -> None:
        """Test with only some hints available."""
        classified = {"order_id": "123"}
        hints = adapter._build_entity_hints(classified)

        assert hints["order_id"] == "123"
        assert len(hints) == 1


class TestOrderIdExtraction:
    """Tests for order ID extraction from forms."""

    def test_extract_from_text(self, adapter: FormAdapter) -> None:
        """Test extraction from text content."""
        order_ids = adapter.extract_order_ids("Order #ORD-12345")
        assert "ORD-12345" in order_ids

    def test_extract_multiple(self, adapter: FormAdapter) -> None:
        """Test extraction of multiple order IDs."""
        # Use order ID formats that match the base extraction patterns
        order_ids = adapter.extract_order_ids("Orders #12345 and order ORD-67890")
        assert "12345" in order_ids
        assert "ORD-67890" in order_ids
