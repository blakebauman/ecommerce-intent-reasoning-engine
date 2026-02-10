"""Email channel adapter with parsing, thread handling, and signature stripping."""

import email
import re
import uuid
from datetime import datetime
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup

from intent_engine.ingestion.base import ChannelAdapter
from intent_engine.models.request import Attachment, InputChannel, IntentRequest


class EmailAdapter(ChannelAdapter):
    """
    Adapter for email channel input.

    Handles:
    - MIME email parsing (multipart, HTML, plain text)
    - Email thread handling (extract most recent message)
    - Signature stripping (common patterns)
    - Quoted reply removal
    - Attachment extraction (metadata only, image analysis deferred)

    Expected input format:
    {
        "raw_email": "<full MIME email string>",
        # OR
        "subject": "Order issue",
        "body": "Where is my order?",
        "from_email": "customer@example.com",
        "from_name": "John Doe",
        "to_email": "support@store.com",
        "message_id": "<abc@mail.com>",
        "in_reply_to": "<xyz@mail.com>",  # optional
        "references": ["<xyz@mail.com>"],  # optional
        "timestamp": "2024-02-09T10:30:00Z",  # optional
        "tenant_id": "merchant-1",
        "attachments": [  # optional
            {"url": "...", "mime_type": "image/png", "filename": "photo.png"}
        ]
    }
    """

    # Common signature delimiters
    SIGNATURE_PATTERNS = [
        r"^--\s*$",  # Standard "--" delimiter
        r"^-{2,}\s*$",  # Multiple dashes
        r"^_{3,}\s*$",  # Underscores
        r"^Sent from my (iPhone|iPad|Android|Samsung|Galaxy|Pixel)",
        r"^Sent from Mail for Windows",
        r"^Get Outlook for",
        r"^Regards,?\s*$",
        r"^Best regards,?\s*$",
        r"^Best,?\s*$",
        r"^Thanks,?\s*$",
        r"^Thank you,?\s*$",
        r"^Cheers,?\s*$",
        r"^Sincerely,?\s*$",
        r"^Kind regards,?\s*$",
        r"^Warm regards,?\s*$",
    ]

    # Quote patterns for thread extraction
    QUOTE_PATTERNS = [
        r"^On .+ wrote:$",  # Gmail style
        r"^On .+, .+ wrote:$",  # Gmail with date
        r"^-{3,} ?Original Message ?-{3,}",  # Outlook
        r"^From: .+$",  # Outlook/other clients
        r"^>+",  # Quoted lines
        r"^\|",  # Pipe-quoted lines
    ]

    @property
    def channel_name(self) -> str:
        return "email"

    def validate(self, raw_input: dict[str, Any]) -> bool:
        """Validate email input structure."""
        # Either raw_email or body+from_email required
        if "raw_email" in raw_input:
            return isinstance(raw_input["raw_email"], str) and len(raw_input["raw_email"]) > 0

        required_fields = ["body", "from_email", "tenant_id"]
        for field in required_fields:
            if field not in raw_input or not raw_input[field]:
                return False

        return True

    async def normalize(self, raw_input: dict[str, Any]) -> IntentRequest:
        """
        Normalize email input to IntentRequest.

        Args:
            raw_input: Raw email data (MIME or structured).

        Returns:
            Normalized IntentRequest.
        """
        if not self.validate(raw_input):
            raise ValueError("Invalid email input: missing required fields")

        # Parse email based on input format
        if "raw_email" in raw_input:
            parsed = self._parse_mime_email(raw_input["raw_email"])
            tenant_id = raw_input.get("tenant_id", "default")
        else:
            parsed = self._parse_structured_email(raw_input)
            tenant_id = raw_input["tenant_id"]

        # Extract the most recent message from thread
        body_text = self._extract_latest_message(parsed["body"])

        # Strip signature
        body_text = self._strip_signature(body_text)

        # Clean and normalize whitespace
        body_text = self._clean_text(body_text)

        # Combine subject and body for raw_text
        raw_text = self._build_raw_text(parsed["subject"], body_text)

        # Generate request ID
        request_id = raw_input.get("request_id", str(uuid.uuid4()))

        # Extract order IDs from both subject and body
        order_ids = self.extract_order_ids(f"{parsed['subject']} {body_text}")

        # Build metadata
        raw_metadata: dict[str, Any] = {
            "email_subject": parsed["subject"],
            "from_email": parsed["from_email"],
            "from_name": parsed["from_name"],
            "to_email": parsed.get("to_email"),
            "message_id": parsed.get("message_id"),
            "in_reply_to": parsed.get("in_reply_to"),
            "references": parsed.get("references", []),
            "is_reply": bool(parsed.get("in_reply_to") or parsed.get("references")),
            "original_body_length": len(parsed["body"]),
            "processed_body_length": len(body_text),
        }

        # Handle attachments
        attachments: list[Attachment] = []
        for att in parsed.get("attachments", []):
            attachments.append(Attachment(
                url=att.get("url", ""),
                mime_type=att.get("mime_type", "application/octet-stream"),
                filename=att.get("filename"),
            ))

        # Determine conversation ID from message threading
        conversation_id = None
        references = parsed.get("references", [])
        if references:
            conversation_id = references[0]  # First reference is typically thread root
        elif parsed.get("in_reply_to"):
            conversation_id = parsed["in_reply_to"]
        elif parsed.get("message_id"):
            conversation_id = parsed["message_id"]

        return IntentRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            channel=InputChannel.EMAIL,
            timestamp=parsed.get("timestamp", datetime.utcnow()),
            raw_text=raw_text,
            raw_metadata=raw_metadata,
            attachments=attachments,
            conversation_id=conversation_id,
            message_index=len(references),  # Approximate position in thread
            customer_id=raw_input.get("customer_id"),
            customer_tier=raw_input.get("customer_tier"),
            order_ids=order_ids + raw_input.get("order_ids", []),
        )

    def _parse_mime_email(self, raw_email: str) -> dict[str, Any]:
        """Parse a raw MIME email string."""
        msg = email.message_from_string(raw_email)

        # Extract headers
        from_addr = parseaddr(msg.get("From", ""))
        to_addr = parseaddr(msg.get("To", ""))

        # Parse timestamp
        timestamp = datetime.utcnow()
        if date_str := msg.get("Date"):
            try:
                timestamp = parsedate_to_datetime(date_str)
            except (ValueError, TypeError):
                pass

        # Extract body
        body = self._extract_body_from_mime(msg)

        # Extract attachments
        attachments = self._extract_attachments_from_mime(msg)

        # Parse references header
        references: list[str] = []
        if ref_header := msg.get("References"):
            references = [r.strip() for r in ref_header.split() if r.strip()]

        return {
            "subject": msg.get("Subject", ""),
            "body": body,
            "from_email": from_addr[1],
            "from_name": from_addr[0],
            "to_email": to_addr[1],
            "message_id": msg.get("Message-ID"),
            "in_reply_to": msg.get("In-Reply-To"),
            "references": references,
            "timestamp": timestamp,
            "attachments": attachments,
        }

    def _extract_body_from_mime(self, msg: EmailMessage | email.message.Message) -> str:
        """Extract text body from MIME message, preferring plain text."""
        body_parts: list[str] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            body_parts.append(payload.decode(charset, errors="replace"))
                        except Exception:
                            body_parts.append(payload.decode("utf-8", errors="replace"))

                elif content_type == "text/html" and not body_parts:
                    # Only use HTML if no plain text found
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            html_text = payload.decode(charset, errors="replace")
                        except Exception:
                            html_text = payload.decode("utf-8", errors="replace")
                        body_parts.append(self._html_to_text(html_text))
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except Exception:
                    text = payload.decode("utf-8", errors="replace")

                if content_type == "text/html":
                    text = self._html_to_text(text)
                body_parts.append(text)

        return "\n".join(body_parts)

    def _extract_attachments_from_mime(
        self, msg: EmailMessage | email.message.Message
    ) -> list[dict[str, Any]]:
        """Extract attachment metadata from MIME message."""
        attachments: list[dict[str, Any]] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in content_disposition or "inline" in content_disposition:
                    filename = part.get_filename()
                    content_type = part.get_content_type()

                    # For now, we just track metadata
                    # Image analysis deferred to Phase 3
                    attachments.append({
                        "mime_type": content_type,
                        "filename": filename,
                        "url": "",  # Would be populated after storage
                    })

        return attachments

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "head", "meta", "link"]):
            element.decompose()

        # Get text with some formatting preservation
        text = soup.get_text(separator="\n")

        # Clean up excessive whitespace
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)

    def _parse_structured_email(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Parse pre-structured email input."""
        timestamp = datetime.utcnow()
        if ts := raw_input.get("timestamp"):
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif isinstance(ts, datetime):
                timestamp = ts

        return {
            "subject": raw_input.get("subject", ""),
            "body": raw_input.get("body", ""),
            "from_email": raw_input.get("from_email", ""),
            "from_name": raw_input.get("from_name", ""),
            "to_email": raw_input.get("to_email"),
            "message_id": raw_input.get("message_id"),
            "in_reply_to": raw_input.get("in_reply_to"),
            "references": raw_input.get("references", []),
            "timestamp": timestamp,
            "attachments": raw_input.get("attachments", []),
        }

    def _extract_latest_message(self, body: str) -> str:
        """
        Extract the most recent message from an email thread.

        Removes quoted replies and previous messages in the thread.
        """
        lines = body.split("\n")
        result_lines: list[str] = []
        in_quote = False

        for line in lines:
            stripped = line.strip()

            # Check if this line starts a quote block
            if self._is_quote_start(stripped):
                in_quote = True
                continue

            # Check if this is a quoted line
            if stripped.startswith(">") or stripped.startswith("|"):
                in_quote = True
                continue

            # Reset quote state if we see non-quote content after empty line
            if in_quote and not stripped:
                # Could be end of quote, but wait for more content
                continue

            if not in_quote:
                result_lines.append(line)

        return "\n".join(result_lines)

    def _is_quote_start(self, line: str) -> bool:
        """Check if a line indicates the start of quoted content."""
        for pattern in self.QUOTE_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False

    def _strip_signature(self, text: str) -> str:
        """Remove email signature from text."""
        lines = text.split("\n")
        result_lines: list[str] = []
        signature_start = len(lines)

        # Find where signature starts by checking patterns
        for i, line in enumerate(lines):
            stripped = line.strip()
            for pattern in self.SIGNATURE_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    signature_start = i
                    break
            if signature_start < len(lines):
                break

        # Keep lines before signature
        result_lines = lines[:signature_start]

        # Also remove trailing empty lines before signature
        while result_lines and not result_lines[-1].strip():
            result_lines.pop()

        return "\n".join(result_lines)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (more than 2 in a row)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _build_raw_text(self, subject: str, body: str) -> str:
        """Combine subject and body into raw_text for processing."""
        if not subject:
            return body

        # Check if subject adds meaningful context
        subject_clean = subject.strip()

        # Remove common subject prefixes
        subject_clean = re.sub(r"^(Re|Fwd|Fw):\s*", "", subject_clean, flags=re.IGNORECASE)
        subject_clean = subject_clean.strip()

        if not subject_clean:
            return body

        # If body is empty, just use subject
        if not body:
            return subject_clean

        # Combine with clear separation
        return f"Subject: {subject_clean}\n\n{body}"
