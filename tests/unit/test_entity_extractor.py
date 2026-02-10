"""Tests for entity extraction."""

import sys

import pytest

# Skip entire module on Python 3.14+ due to spaCy incompatibility
if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy not compatible with Python 3.14+ (uses Pydantic v1)",
        allow_module_level=True,
    )

from intent_engine.extractors.entity_extractor import EntityExtractor
from intent_engine.models.entity import EntityType


@pytest.fixture
def extractor() -> EntityExtractor:
    """Create an entity extractor instance."""
    return EntityExtractor()


class TestOrderIdExtraction:
    """Tests for order ID extraction."""

    def test_extract_order_id_with_hash(self, extractor: EntityExtractor) -> None:
        """Test extracting order ID with # prefix."""
        result = extractor.extract("Where is my order #12345?")
        order_ids = [e for e in result.entities if e.entity_type == EntityType.ORDER_ID]
        assert len(order_ids) == 1
        assert order_ids[0].value == "12345"

    def test_extract_order_id_ord_format(self, extractor: EntityExtractor) -> None:
        """Test extracting order ID in ORD-XXXXX format."""
        result = extractor.extract("Track order ORD-98765 please")
        order_ids = [e for e in result.entities if e.entity_type == EntityType.ORDER_ID]
        assert len(order_ids) >= 1
        assert any("98765" in oid.value for oid in order_ids)

    def test_extract_multiple_order_ids(self, extractor: EntityExtractor) -> None:
        """Test extracting multiple order IDs."""
        result = extractor.extract("Check orders #11111 and #22222")
        order_ids = [e for e in result.entities if e.entity_type == EntityType.ORDER_ID]
        assert len(order_ids) == 2

    def test_no_order_id(self, extractor: EntityExtractor) -> None:
        """Test when no order ID is present."""
        result = extractor.extract("Where is my order?")
        order_ids = [e for e in result.entities if e.entity_type == EntityType.ORDER_ID]
        assert len(order_ids) == 0


class TestTrackingNumberExtraction:
    """Tests for tracking number extraction."""

    def test_extract_ups_tracking(self, extractor: EntityExtractor) -> None:
        """Test extracting UPS tracking number."""
        result = extractor.extract("Tracking: 1Z999AA10123456784")
        tracking = [e for e in result.entities if e.entity_type == EntityType.TRACKING_NUMBER]
        assert len(tracking) >= 1


class TestSizeExtraction:
    """Tests for size extraction."""

    def test_extract_letter_size(self, extractor: EntityExtractor) -> None:
        """Test extracting letter sizes (S, M, L, etc)."""
        result = extractor.extract("I need size L please")
        sizes = [e for e in result.entities if e.entity_type == EntityType.SIZE]
        assert len(sizes) == 1
        assert sizes[0].value.upper() == "L"

    def test_extract_word_size(self, extractor: EntityExtractor) -> None:
        """Test extracting word sizes (small, medium, large)."""
        result = extractor.extract("I want the medium one")
        sizes = [e for e in result.entities if e.entity_type == EntityType.SIZE]
        assert len(sizes) == 1


class TestColorExtraction:
    """Tests for color extraction."""

    def test_extract_color(self, extractor: EntityExtractor) -> None:
        """Test extracting colors."""
        result = extractor.extract("I want the blue one instead")
        colors = [e for e in result.entities if e.entity_type == EntityType.COLOR]
        assert len(colors) == 1
        assert colors[0].value.lower() == "blue"


class TestDeadlineExtraction:
    """Tests for deadline extraction."""

    def test_extract_by_day(self, extractor: EntityExtractor) -> None:
        """Test extracting deadline with day name."""
        result = extractor.extract("I need this by Friday")
        deadlines = [e for e in result.entities if e.entity_type == EntityType.DEADLINE]
        assert len(deadlines) >= 1

    def test_extract_within_days(self, extractor: EntityExtractor) -> None:
        """Test extracting deadline with 'within X days'."""
        result = extractor.extract("Deliver within 3 days")
        deadlines = [e for e in result.entities if e.entity_type == EntityType.DEADLINE]
        assert len(deadlines) >= 1

    def test_extract_urgent(self, extractor: EntityExtractor) -> None:
        """Test extracting urgent deadline."""
        result = extractor.extract("This is urgent!")
        deadlines = [e for e in result.entities if e.entity_type == EntityType.DEADLINE]
        assert len(deadlines) >= 1


class TestReasonExtraction:
    """Tests for reason extraction."""

    def test_extract_damaged_reason(self, extractor: EntityExtractor) -> None:
        """Test extracting damaged reason."""
        result = extractor.extract("The item is damaged")
        reasons = [e for e in result.entities if e.entity_type == EntityType.REASON]
        assert len(reasons) >= 1
        assert any("damaged" in r.value for r in reasons)

    def test_extract_wrong_size_reason(self, extractor: EntityExtractor) -> None:
        """Test extracting wrong size reason."""
        result = extractor.extract("It doesn't fit, wrong size")
        reasons = [e for e in result.entities if e.entity_type == EntityType.REASON]
        assert len(reasons) >= 1


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_extract_order_ids_method(self, extractor: EntityExtractor) -> None:
        """Test the extract_order_ids convenience method."""
        order_ids = extractor.extract_order_ids("Check order #12345 and ORD-67890")
        assert len(order_ids) >= 2
