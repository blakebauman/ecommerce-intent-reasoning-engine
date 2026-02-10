"""Tests for email channel adapter."""

import pytest
from datetime import datetime

from intent_engine.ingestion.email import EmailAdapter
from intent_engine.models.request import InputChannel


@pytest.fixture
def adapter() -> EmailAdapter:
    """Create an email adapter instance."""
    return EmailAdapter()


class TestEmailAdapterValidation:
    """Tests for email input validation."""

    def test_valid_structured_email(self, adapter: EmailAdapter) -> None:
        """Test validation of valid structured email."""
        raw_input = {
            "body": "Where is my order?",
            "from_email": "customer@example.com",
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is True

    def test_valid_mime_email(self, adapter: EmailAdapter) -> None:
        """Test validation of valid MIME email."""
        raw_input = {
            "raw_email": "From: test@example.com\nSubject: Help\n\nBody text",
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is True

    def test_invalid_missing_body(self, adapter: EmailAdapter) -> None:
        """Test validation fails without body."""
        raw_input = {
            "from_email": "customer@example.com",
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is False

    def test_invalid_missing_from_email(self, adapter: EmailAdapter) -> None:
        """Test validation fails without from_email."""
        raw_input = {
            "body": "Where is my order?",
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is False

    def test_invalid_missing_tenant(self, adapter: EmailAdapter) -> None:
        """Test validation fails without tenant_id."""
        raw_input = {
            "body": "Where is my order?",
            "from_email": "customer@example.com",
        }
        assert adapter.validate(raw_input) is False

    def test_invalid_empty_body(self, adapter: EmailAdapter) -> None:
        """Test validation fails with empty body."""
        raw_input = {
            "body": "",
            "from_email": "customer@example.com",
            "tenant_id": "merchant-1",
        }
        assert adapter.validate(raw_input) is False


class TestEmailAdapterNormalization:
    """Tests for email normalization."""

    @pytest.mark.asyncio
    async def test_normalize_simple_email(self, adapter: EmailAdapter) -> None:
        """Test normalization of simple email."""
        raw_input = {
            "body": "Where is my order #ORD-12345?",
            "subject": "Order question",
            "from_email": "customer@example.com",
            "from_name": "John Doe",
            "tenant_id": "merchant-1",
        }

        request = await adapter.normalize(raw_input)

        assert request.channel == InputChannel.EMAIL
        assert request.tenant_id == "merchant-1"
        assert "ORD-12345" in request.raw_text
        assert "Order question" in request.raw_text
        assert request.raw_metadata["from_email"] == "customer@example.com"
        assert request.raw_metadata["from_name"] == "John Doe"
        assert "ORD-12345" in request.order_ids

    @pytest.mark.asyncio
    async def test_normalize_reply_email(self, adapter: EmailAdapter) -> None:
        """Test normalization of reply email with metadata."""
        raw_input = {
            "body": "Thanks for the update",
            "subject": "Re: Order question",
            "from_email": "customer@example.com",
            "tenant_id": "merchant-1",
            "message_id": "<abc123@mail.com>",
            "in_reply_to": "<xyz789@mail.com>",
            "references": ["<xyz789@mail.com>"],
        }

        request = await adapter.normalize(raw_input)

        assert request.raw_metadata["is_reply"] is True
        assert request.conversation_id == "<xyz789@mail.com>"
        assert request.message_index == 1  # One reference

    @pytest.mark.asyncio
    async def test_normalize_with_timestamp(self, adapter: EmailAdapter) -> None:
        """Test normalization preserves timestamp."""
        raw_input = {
            "body": "Test message",
            "from_email": "customer@example.com",
            "tenant_id": "merchant-1",
            "timestamp": "2024-02-09T10:30:00Z",
        }

        request = await adapter.normalize(raw_input)

        assert request.timestamp.year == 2024
        assert request.timestamp.month == 2
        assert request.timestamp.day == 9

    @pytest.mark.asyncio
    async def test_normalize_invalid_raises(self, adapter: EmailAdapter) -> None:
        """Test normalization raises for invalid input."""
        raw_input = {"body": "", "tenant_id": "test"}

        with pytest.raises(ValueError, match="Invalid email input"):
            await adapter.normalize(raw_input)


class TestSignatureStripping:
    """Tests for email signature removal."""

    def test_strip_standard_delimiter(self, adapter: EmailAdapter) -> None:
        """Test stripping with -- delimiter."""
        text = "Hello, I need help with my order.\n\n--\nJohn Doe\nCustomer"
        result = adapter._strip_signature(text)
        assert "John Doe" not in result
        assert "I need help" in result

    def test_strip_regards_signature(self, adapter: EmailAdapter) -> None:
        """Test stripping 'Regards' signature."""
        text = "Please help me.\n\nRegards,\nJohn"
        result = adapter._strip_signature(text)
        assert "John" not in result
        assert "Please help" in result

    def test_strip_thanks_signature(self, adapter: EmailAdapter) -> None:
        """Test stripping 'Thanks' signature."""
        text = "I have a question.\n\nThanks,\nSarah"
        result = adapter._strip_signature(text)
        assert "Sarah" not in result
        assert "I have a question" in result

    def test_strip_sent_from_iphone(self, adapter: EmailAdapter) -> None:
        """Test stripping mobile signature."""
        text = "Help me with order\n\nSent from my iPhone"
        result = adapter._strip_signature(text)
        assert "iPhone" not in result
        assert "Help me" in result

    def test_no_signature(self, adapter: EmailAdapter) -> None:
        """Test text without signature is preserved."""
        text = "Hello, I need help with my order."
        result = adapter._strip_signature(text)
        assert result == text


class TestThreadExtraction:
    """Tests for email thread handling."""

    def test_extract_from_quoted_reply(self, adapter: EmailAdapter) -> None:
        """Test extraction removes quoted content."""
        body = """I have another question about this.

On Mon, Jan 15, 2024, Support wrote:
> Thanks for contacting us.
> Your order is on the way."""
        result = adapter._extract_latest_message(body)
        assert "another question" in result
        assert "Thanks for contacting" not in result

    def test_extract_from_outlook_reply(self, adapter: EmailAdapter) -> None:
        """Test extraction removes Outlook-style quotes."""
        body = """My response here.

----- Original Message -----
From: Support
Previous message content."""
        result = adapter._extract_latest_message(body)
        assert "My response here" in result
        assert "Previous message" not in result

    def test_extract_removes_angle_bracket_quotes(self, adapter: EmailAdapter) -> None:
        """Test extraction removes > quoted lines."""
        body = """New content

> Quoted line 1
> Quoted line 2"""
        result = adapter._extract_latest_message(body)
        assert "New content" in result
        assert "Quoted line" not in result

    def test_extract_plain_message(self, adapter: EmailAdapter) -> None:
        """Test plain message without quotes is preserved."""
        body = "Just a simple message without any quotes."
        result = adapter._extract_latest_message(body)
        assert result.strip() == body


class TestHtmlToText:
    """Tests for HTML to text conversion."""

    def test_simple_html(self, adapter: EmailAdapter) -> None:
        """Test simple HTML conversion."""
        html = "<p>Hello</p><p>World</p>"
        result = adapter._html_to_text(html)
        assert "Hello" in result
        assert "World" in result

    def test_removes_script_tags(self, adapter: EmailAdapter) -> None:
        """Test script tags are removed."""
        html = "<p>Content</p><script>alert('bad')</script>"
        result = adapter._html_to_text(html)
        assert "Content" in result
        assert "alert" not in result

    def test_removes_style_tags(self, adapter: EmailAdapter) -> None:
        """Test style tags are removed."""
        html = "<style>.class{color:red}</style><p>Text</p>"
        result = adapter._html_to_text(html)
        assert "Text" in result
        assert "color" not in result


class TestMimeEmailParsing:
    """Tests for MIME email parsing."""

    def test_parse_simple_mime(self, adapter: EmailAdapter) -> None:
        """Test parsing simple MIME email."""
        raw_email = """From: John Doe <john@example.com>
To: support@store.com
Subject: Order Help
Date: Mon, 15 Jan 2024 10:30:00 -0000
Message-ID: <test123@mail.com>

Where is my order?"""

        parsed = adapter._parse_mime_email(raw_email)

        assert parsed["from_email"] == "john@example.com"
        assert parsed["from_name"] == "John Doe"
        assert parsed["subject"] == "Order Help"
        assert "Where is my order" in parsed["body"]
        assert parsed["message_id"] == "<test123@mail.com>"

    def test_parse_reply_headers(self, adapter: EmailAdapter) -> None:
        """Test parsing reply headers."""
        raw_email = """From: john@example.com
To: support@store.com
Subject: Re: Order Help
In-Reply-To: <original@mail.com>
References: <original@mail.com> <reply1@mail.com>

Thanks for the help."""

        parsed = adapter._parse_mime_email(raw_email)

        assert parsed["in_reply_to"] == "<original@mail.com>"
        assert len(parsed["references"]) == 2


class TestRawTextBuilding:
    """Tests for combining subject and body."""

    def test_combines_subject_and_body(self, adapter: EmailAdapter) -> None:
        """Test subject and body combination."""
        result = adapter._build_raw_text("Order Problem", "My order is late.")
        assert "Subject: Order Problem" in result
        assert "My order is late" in result

    def test_removes_re_prefix(self, adapter: EmailAdapter) -> None:
        """Test Re: prefix is removed."""
        result = adapter._build_raw_text("Re: Order Help", "Thanks")
        assert "Re:" not in result
        assert "Order Help" in result

    def test_empty_subject(self, adapter: EmailAdapter) -> None:
        """Test empty subject just returns body."""
        result = adapter._build_raw_text("", "Just the body text")
        assert result == "Just the body text"

    def test_empty_body(self, adapter: EmailAdapter) -> None:
        """Test empty body just returns subject."""
        result = adapter._build_raw_text("Important Subject", "")
        assert result == "Important Subject"


class TestOrderIdExtraction:
    """Tests for order ID extraction from emails."""

    def test_extract_from_subject_and_body(self, adapter: EmailAdapter) -> None:
        """Test extraction from both subject and body."""
        order_ids = adapter.extract_order_ids("Order #ORD-12345 - also see ORD-67890")
        assert "ORD-12345" in order_ids
        assert "ORD-67890" in order_ids

    def test_extract_shopify_format(self, adapter: EmailAdapter) -> None:
        """Test extraction of Shopify order format."""
        order_ids = adapter.extract_order_ids("My order #1234 is late")
        assert "1234" in order_ids

    def test_no_order_ids(self, adapter: EmailAdapter) -> None:
        """Test no extraction when no order IDs."""
        order_ids = adapter.extract_order_ids("I have a general question")
        assert len(order_ids) == 0
