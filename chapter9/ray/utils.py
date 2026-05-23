"""Utility functions."""
import re
import html

def clean_html_content(text: str, max_length: int = 200) -> str:
    """Clean HTML content to make it readable for display.
    
    Args:
        text: Raw text content that may contain HTML tags and entities.
        max_length: Maximum length of cleaned text to return.
        
    Returns:
        Cleaned, readable text without HTML tags or entities.
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (like &#x27; -> ')
    text = html.unescape(text)
    
    # Clean up whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text