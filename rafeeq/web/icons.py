"""Inline SVG icons (no icon font, no CDN). Usage in templates: {{ icon("send") }}."""
from markupsafe import Markup

_PATHS = {
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6"/>',
    "arrow": '<path d="M19 12H5M12 19l-7-7 7-7"/>',
    "chevron": '<path d="m15 18-6-6 6-6"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "book": '<path d="M12 7v14M3 5.5A2.5 2.5 0 0 1 5.5 3H9a3 3 0 0 1 3 3v15a3 3 0 0 0-3-3H5.5A2.5 2.5 0 0 0 3 20.5zM21 5.5A2.5 2.5 0 0 0 18.5 3H15a3 3 0 0 0-3 3v15a3 3 0 0 1 3-3h3.5a2.5 2.5 0 0 1 2.5 2.5z"/>',
    "dashboard": '<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="10" cy="7" r="4"/><path d="M20 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M8 15v-3M13 15V6M18 15V9"/>',
    "sparkle": '<path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3zM19 14l1 2.5 2.5 1-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/>',
    "chat": '<path d="M21 11.5a8.5 8.5 0 0 1-12.3 7.6L3 21l1.9-5.1A8.5 8.5 0 1 1 21 11.5z"/>',
    "clip": '<path d="m21 11.5-8.5 8.5a5 5 0 0 1-7.1-7.1l9.2-9.2a3.5 3.5 0 0 1 4.9 4.9l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/>',
    "presentation": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M12 17v4M8 21h8M8 9h8M8 12h4"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
    "check": '<path d="m5 12 5 5L20 7"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
    "x_circle": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
    "alert": '<path d="m10.3 3.9-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3.1l-8-14a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "bulb": '<path d="M9 18h6M10 22h4M8 14a6 6 0 1 1 8 0c-.8.6-1 1.2-1 2H9c0-.8-.2-1.4-1-2z"/>',
    "lock": '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    "logout": '<path d="M10 17l5-5-5-5M15 12H3M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6"/>',
    "door": '<path d="M4 21h16M6 21V3h12v18M14 12h.01"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5M4 17v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5M4 17v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "badge": '<path d="M12 3 15 5l4 .5.5 4 2 2.5-2 2.5-.5 4-4 .5-3 2-3-2-4-.5-.5-4-2-2.5 2-2.5.5-4L9 5z"/><path d="m8.5 12 2.2 2.2 4.8-4.8"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
    "send": '<path d="m22 2-7 20-4-9-9-4zM22 2 11 13"/>',
    "home": '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1z"/>',
    "quiz": '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2h6v2M8 12l2.5 2.5L16 9"/>',
    "question": '<circle cx="12" cy="12" r="9"/><path d="M9.8 9a2.3 2.3 0 1 1 3.8 1.7c-1 .8-1.6 1.2-1.6 2.3M12 16h.01"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"/>',
    "trend": '<path d="M3 3v18h18M6 15l4-4 4 2 5-7"/>',
    "pen": '<path d="m16 4 4 4M4 20l4-.8L19 8a2.8 2.8 0 0 0-4-4L4 15z"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "code": '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-2.6-6.4L21 8"/><path d="M21 3v5h-5"/>',
    "key": '<circle cx="7.5" cy="15.5" r="4.5"/><path d="m10.7 12.3 9.3-9.3M17 6l3 3M14 9l2 2"/>',
}


def icon(name: str, size: int = 20, cls: str = "") -> Markup:
    body = _PATHS.get(name, _PATHS["info"])
    return Markup(
        f'<svg class="icon {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true" focusable="false">{body}</svg>'
    )
