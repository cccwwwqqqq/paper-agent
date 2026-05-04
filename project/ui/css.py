custom_css = """
    :root {
        --page-bg: #ffffff;
        --surface-bg: #ffffff;
        --surface-muted: #f8fafc;
        --surface-subtle: #f3f4f6;
        --border-color: #d1d5db;
        --border-color-strong: #cbd5e1;
        --text-color: #111827;
        --text-muted: #6b7280;
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --danger-color: #dc2626;
        --danger-hover: #b91c1c;
    }

    /* ============================================
       MAIN CONTAINER
       ============================================ */
    .progress-text { 
        display: none !important;
    }

    html,
    body {
        background: var(--page-bg) !important;
        color: var(--text-color) !important;
    }
    
    .gradio-container { 
        max-width: 1000px !important;
        width: 100% !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        background: var(--page-bg) !important;
        color: var(--text-color) !important;
    }

    .gradio-container,
    .gradio-container *,
    [data-testid="block-wrapper"] {
        color: var(--text-color);
    }
    
    /* ============================================
       TABS
       ============================================ */
    button[role="tab"] {
        color: var(--text-muted) !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        transition: all 0.2s ease !important;
        background: transparent !important;
    }
    
    button[role="tab"]:hover {
        color: var(--text-color) !important;
    }
    
    button[role="tab"][aria-selected="true"] {
        color: var(--text-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
        border-radius: 0 !important;
        background: transparent !important;
    }
    
    .tabs {
        border-bottom: none !important;
        border-radius: 0 !important;
    }
    
    .tab-nav {
        border-bottom: 1px solid var(--border-color) !important;
        border-radius: 0 !important;
    }
    
    button[role="tab"]::before,
    button[role="tab"]::after,
    .tabs::before,
    .tabs::after,
    .tab-nav::before,
    .tab-nav::after {
        display: none !important;
        content: none !important;
        border-radius: 0 !important;
    }
    
    #doc-management-tab {
        max-width: 500px !important;
        margin: 0 auto !important;
    }
    
    /* ============================================
       BUTTONS
       ============================================ */
    button {
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    
    .primary {
        background: var(--primary-color) !important;
        color: white !important;
    }
    
    .primary:hover {
        background: var(--primary-hover) !important;
        transform: translateY(-1px) !important;
    }
    
    .stop {
        background: var(--danger-color) !important;
        color: white !important;
    }
    
    .stop:hover {
        background: var(--danger-hover) !important;
        transform: translateY(-1px) !important;
    }
    
    /* ============================================
       CHAT INPUT BOX
       ============================================ */
    textarea[placeholder="Type a message..."],
    textarea[placeholder="请输入你的问题..."],
    textarea[data-testid*="textbox"]:not(#file-list-box textarea) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--text-color) !important;
    }
    
    textarea[placeholder="Type a message..."]:focus,
    textarea[placeholder="请输入你的问题..."]:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .gr-text-input:has(textarea[placeholder="Type a message..."]),
    .gr-text-input:has(textarea[placeholder="请输入你的问题..."]),
    [class*="chatbot"] + * [data-testid="textbox"],
    form:has(textarea[placeholder="Type a message..."]) > div,
    form:has(textarea[placeholder="请输入你的问题..."]) > div {
        background: transparent !important;
        border: none !important;
        gap: 12px !important;
    }
    
    form:has(textarea[placeholder="Type a message..."]) button,
    form:has(textarea[placeholder="请输入你的问题..."]) button,
    [class*="chatbot"] ~ * button[type="submit"] {
        background: transparent !important;
        border: none !important;
        padding: 8px !important;
    }
    
    form:has(textarea[placeholder="Type a message..."]) button:hover,
    form:has(textarea[placeholder="请输入你的问题..."]) button:hover {
        background: rgba(37, 99, 235, 0.08) !important;
    }
    
    form:has(textarea[placeholder="Type a message..."]),
    form:has(textarea[placeholder="请输入你的问题..."]) {
        gap: 12px !important;
        display: flex !important;
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 8px 10px !important;
    }
    
    /* ============================================
       FILE UPLOAD
       ============================================ */
    .file-preview, 
    [data-testid="file-upload"] {
        background: var(--surface-muted) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 5px !important;
        color: var(--text-color) !important;
        min-height: 200px !important;
    }
    
    .file-preview:hover, 
    [data-testid="file-upload"]:hover {
        border-color: var(--primary-color) !important;
        background: #eff6ff !important;
    }
    
    .file-preview *,
    [data-testid="file-upload"] * {
        color: var(--text-color) !important;
    }
    
    .file-preview .label,
    [data-testid="file-upload"] .label {
        display: none !important;
    }
    
    /* ============================================
       INPUTS & TEXTAREAS
       ============================================ */
    input, 
    textarea {
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-color) !important;
        transition: border-color 0.2s ease !important;
    }
    
    input:focus, 
    textarea:focus {
        border-color: var(--primary-color) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    textarea[readonly] {
        background: var(--surface-muted) !important;
        color: var(--text-muted) !important;
    }
    
    /* ============================================
       FILE LIST BOX
       ============================================ */
    #file-list-box {
        background: var(--surface-muted) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    
    #file-list-box textarea {
        background: transparent !important;
        border: none !important;
        color: var(--text-color) !important;
        padding: 0 !important;
    }
    
    /* ============================================
       CHATBOT CONTAINER
       ============================================ */
    .chatbot {
        border-radius: 12px !important;
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-color) !important;
    }

    .chatbot .message-wrap,
    .chatbot > div {
        gap: 8px !important;
        padding: 12px !important;
        background: var(--surface-bg) !important;
    }

    .chatbot .bubble-wrap,
    .chatbot .message-row,
    .chatbot .message-wrap {
        background: transparent !important;
    }

    .chatbot .placeholder,
    .chatbot .placeholder *,
    .chatbot [class*="placeholder"] {
        color: var(--text-muted) !important;
    }

    /* ============================================
       MESSAGE BUBBLES
       ============================================ */
    .message {
        border-radius: 10px !important;
    }

    .message.user {
        background: var(--primary-color) !important;
        color: white !important;
    }
    
    .message.bot {
        background: var(--surface-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color-strong) !important;
        width: fit-content !important;
        max-width: 90% !important;
    }

    .message.bot *,
    .message.bot p,
    .message.bot li,
    .message.bot strong,
    .message.bot span,
    .message.bot code,
    .message.bot pre {
        color: var(--text-color) !important;
    }

    .message.bot pre,
    .message.bot code {
        background: var(--surface-subtle) !important;
        border-radius: 8px !important;
    }
    
    .message-row img {
        margin: 0px !important;
    }

    .avatar-container img {
        padding: 0px !important;
    }

    /* ============================================
       PROGRESS BAR
       ============================================ */
    .progress-bar-wrap {
        border-radius: 10px !important;
        overflow: hidden !important;
        background: var(--surface-subtle) !important;
    }

    .progress-bar {
        border-radius: 10px !important;
        background: var(--primary-color) !important;
    }
    
    /* ============================================
       TYPOGRAPHY
       ============================================ */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }

    p,
    label,
    .prose,
    .prose *,
    .gr-markdown,
    .gr-markdown * {
        color: var(--text-color) !important;
    }

    .info,
    .gradio-container .description,
    .gradio-container .hint,
    .gradio-container .secondary-text {
        color: var(--text-muted) !important;
    }
    
    /* ============================================
       GLOBAL OVERRIDES
       ============================================ */
    * {
        box-shadow: none !important;
    }
    
    footer {
        visibility: hidden;
    }
"""
