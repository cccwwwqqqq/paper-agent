custom_css = """
    :root {
        --page-bg: linear-gradient(180deg, #f5f7fb 0%, #eef2f8 100%);
        --surface-bg: rgba(255, 255, 255, 0.92);
        --surface-strong: #ffffff;
        --surface-soft: #f8fafc;
        --surface-ink: #eef4ff;
        --border-color: rgba(148, 163, 184, 0.28);
        --border-strong: rgba(99, 102, 241, 0.22);
        --text-color: #0f172a;
        --text-muted: #5b6476;
        --headline-color: #0b1220;
        --primary-color: #3158d6;
        --primary-hover: #2749bb;
        --danger-color: #c93b2d;
        --danger-hover: #a92f23;
        --shadow-soft: 0 18px 50px rgba(15, 23, 42, 0.08);
        --shadow-card: 0 12px 32px rgba(15, 23, 42, 0.06);
        --radius-xl: 24px;
        --radius-lg: 18px;
        --radius-md: 14px;
    }

    html,
    body {
        background: var(--page-bg) !important;
        color: var(--text-color) !important;
    }

    body {
        background-attachment: fixed !important;
    }

    .progress-text {
        display: none !important;
    }

    .gradio-container {
        max-width: 1180px !important;
        width: min(1180px, calc(100vw - 32px)) !important;
        margin: 0 auto !important;
        padding: 24px 0 40px !important;
        font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
        color: var(--text-color) !important;
        background: transparent !important;
    }

    .gradio-container,
    .gradio-container * {
        box-shadow: none !important;
    }

    .app-shell {
        gap: 22px !important;
    }

    .app-hero {
        padding: 28px 30px 24px;
        border: 1px solid rgba(99, 102, 241, 0.14);
        border-radius: 30px;
        background:
            radial-gradient(circle at top right, rgba(49, 88, 214, 0.14), transparent 34%),
            radial-gradient(circle at left bottom, rgba(14, 165, 233, 0.10), transparent 28%),
            rgba(255, 255, 255, 0.86);
        backdrop-filter: blur(14px);
        box-shadow: var(--shadow-soft) !important;
    }

    .app-eyebrow {
        margin: 0 0 10px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #4960a8 !important;
    }

    .app-hero h1 {
        margin: 0 !important;
        font-size: 34px !important;
        line-height: 1.1 !important;
        color: var(--headline-color) !important;
    }

    .app-subtitle {
        margin: 12px 0 0 !important;
        max-width: 760px;
        font-size: 15px !important;
        line-height: 1.75 !important;
        color: var(--text-muted) !important;
    }

    .workspace-grid,
    .content-grid {
        gap: 18px !important;
    }

    .panel {
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-xl) !important;
        background: var(--surface-bg) !important;
        box-shadow: var(--shadow-card) !important;
        padding: 22px !important;
        backdrop-filter: blur(12px);
        gap: 14px !important;
    }

    .panel-strong {
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(245, 248, 255, 0.92)) !important;
    }

    .panel-muted {
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(248, 250, 252, 0.92)) !important;
    }

    .panel-head {
        margin-bottom: 2px !important;
    }

    .panel-head h2,
    .panel-head h3 {
        margin: 0 !important;
        color: var(--headline-color) !important;
        letter-spacing: -0.02em;
    }

    .panel-head h2 {
        font-size: 28px !important;
    }

    .panel-head h3 {
        font-size: 20px !important;
    }

    .panel-head p {
        margin: 8px 0 0 !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
        color: var(--text-muted) !important;
    }

    .panel-head-inline {
        margin-bottom: 6px !important;
    }

    .workspace-tabs {
        gap: 18px !important;
    }

    .tab-nav {
        margin: 4px 0 2px !important;
        padding: 0 4px 8px !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.22) !important;
    }

    button[role="tab"] {
        min-height: 42px !important;
        padding: 0 16px !important;
        border-radius: 999px !important;
        color: var(--text-muted) !important;
        background: transparent !important;
        border: none !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }

    button[role="tab"]:hover {
        color: var(--headline-color) !important;
        background: rgba(49, 88, 214, 0.08) !important;
    }

    button[role="tab"][aria-selected="true"] {
        color: #16358f !important;
        background: rgba(49, 88, 214, 0.12) !important;
        border-bottom: none !important;
    }

    button {
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        transition: transform 0.16s ease, background 0.16s ease, opacity 0.16s ease !important;
    }

    button:hover {
        transform: translateY(-1px) !important;
    }

    .primary,
    .action-button {
        background: linear-gradient(135deg, var(--primary-color), #4e6ff2) !important;
        color: #ffffff !important;
    }

    .primary:hover,
    .action-button:hover {
        background: linear-gradient(135deg, var(--primary-hover), #3f60dc) !important;
    }

    .secondary-action {
        background: #e7ebf3 !important;
        color: #24324d !important;
    }

    .secondary-action:hover {
        background: #dbe3f0 !important;
    }

    .stop,
    .danger-action {
        background: linear-gradient(135deg, var(--danger-color), #de584a) !important;
        color: #ffffff !important;
    }

    .stop:hover,
    .danger-action:hover {
        background: linear-gradient(135deg, var(--danger-hover), #ca4437) !important;
    }

    .action-row {
        gap: 14px !important;
        margin-top: 4px !important;
    }

    .action-row > * {
        flex: 1 1 0 !important;
    }

    .compact-input textarea,
    .compact-input input,
    .compact-output textarea,
    .compact-output input,
    .compact-select input,
    .compact-select textarea,
    .compact-select button,
    input,
    textarea {
        border-radius: var(--radius-md) !important;
        border: 1px solid rgba(148, 163, 184, 0.30) !important;
        background: var(--surface-strong) !important;
        color: var(--text-color) !important;
        min-height: 52px !important;
    }

    input,
    textarea {
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
    }

    input:focus,
    textarea:focus {
        border-color: rgba(49, 88, 214, 0.55) !important;
        box-shadow: 0 0 0 4px rgba(49, 88, 214, 0.10) !important;
        outline: none !important;
    }

    .compact-input,
    .compact-output,
    .compact-select {
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .compact-input label,
    .compact-output label,
    .compact-select label {
        margin-bottom: 4px !important;
        font-weight: 700 !important;
        color: var(--headline-color) !important;
    }

    .field-meta {
        margin: 0 0 8px !important;
    }

    .field-meta label {
        display: block !important;
        margin: 0 !important;
        font-weight: 700 !important;
        color: var(--headline-color) !important;
        font-size: 15px !important;
        line-height: 1.4 !important;
    }

    .field-meta p {
        margin: 6px 0 0 !important;
        color: var(--text-muted) !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .compact-output > *,
    .compact-select > * {
        margin: 0 !important;
    }

    .compact-output .wrap,
    .compact-select .wrap,
    .compact-output .block,
    .compact-select .block {
        margin: 0 !important;
        padding: 0 !important;
        min-height: auto !important;
    }

    .compact-select .info,
    .compact-select .description,
    .compact-select [class*="info"],
    .compact-select [class*="description"] {
        display: none !important;
    }

    .compact-output textarea[readonly] {
        background: var(--surface-soft) !important;
        color: var(--headline-color) !important;
    }

    .panel-upload,
    .panel-library {
        min-height: 100% !important;
    }

    #doc-management-tab {
        max-width: none !important;
        margin: 0 !important;
    }

    .panel-upload .wrap,
    .panel-library .wrap {
        gap: 16px !important;
    }

    [data-testid="file-upload"],
    .file-preview {
        min-height: 240px !important;
        border: 1px dashed rgba(99, 102, 241, 0.24) !important;
        border-radius: 20px !important;
        background:
            linear-gradient(180deg, rgba(243, 247, 255, 0.88), rgba(255, 255, 255, 0.92)) !important;
    }

    [data-testid="file-upload"]:hover,
    .file-preview:hover {
        border-color: rgba(49, 88, 214, 0.45) !important;
        background:
            linear-gradient(180deg, rgba(236, 243, 255, 0.96), rgba(255, 255, 255, 0.96)) !important;
    }

    [data-testid="file-upload"] * ,
    .file-preview * {
        color: var(--text-color) !important;
    }

    #file-list-box {
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        border-radius: 20px !important;
        background: var(--surface-soft) !important;
        padding: 14px !important;
        min-height: 300px !important;
    }

    #file-list-box textarea {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--text-color) !important;
        line-height: 1.7 !important;
        padding: 0 !important;
    }

    .chat-shell {
        gap: 16px !important;
    }

    .chat-surface,
    .chatbot {
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 24px !important;
        background: rgba(255, 255, 255, 0.88) !important;
        overflow: hidden !important;
    }

    .chatbot .message-wrap,
    .chatbot > div {
        padding: 18px !important;
        gap: 16px !important;
        background: transparent !important;
    }

    .chatbot .avatar-container,
    .chatbot .avatar-container img,
    .chatbot .message-row > img {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        min-width: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .message {
        border: none !important;
        box-shadow: none !important;
    }

    .message.user {
        background: linear-gradient(135deg, #3158d6, #4f6ff1) !important;
        border: none !important;
        box-shadow: none !important;
        color: #ffffff !important;
        width: fit-content !important;
        max-width: min(72%, 760px) !important;
        min-width: 0 !important;
        margin-left: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        padding: 12px 16px !important;
        line-height: 1.6 !important;
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        border-radius: 18px !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        position: relative !important;
        padding-right: 50px !important;
    }

    .message.user > *,
    .message.user .prose,
    .message.user .gr-markdown,
    .message.user [data-testid="markdown"],
    .message.user .message-content,
    .message.user .md,
    .message.user .text,
    .message.user .component-wrap,
    .message.user .message-markdown {
        display: inline !important;
        width: auto !important;
        max-width: none !important;
        min-width: 0 !important;
        flex: 0 0 auto !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }

    .message.user p,
    .message.user span,
    .message.user strong,
    .message.user em,
    .message.user li,
    .message.user code {
        color: #ffffff !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: inline !important;
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
        word-break: normal !important;
    }

    .message.user button,
    .message.user [role="button"],
    .message.user [class*="icon-button"],
    .message.user [aria-label*="copy"],
    .message.user [aria-label*="复制"] {
        position: absolute !important;
        right: 12px !important;
        bottom: 10px !important;
        width: 26px !important;
        height: 26px !important;
        min-width: 26px !important;
        min-height: 26px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(255, 255, 255, 0.18) !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        color: #ffffff !important;
        z-index: 2 !important;
    }

    .message.user button:hover,
    .message.user [role="button"]:hover,
    .message.user [class*="icon-button"]:hover,
    .message.user [aria-label*="copy"]:hover,
    .message.user [aria-label*="复制"]:hover {
        background: rgba(255, 255, 255, 0.26) !important;
    }

    .message.user button svg,
    .message.user [role="button"] svg,
    .message.user [class*="icon-button"] svg,
    .message.user [aria-label*="copy"] svg,
    .message.user [aria-label*="复制"] svg {
        width: 15px !important;
        height: 15px !important;
        min-width: 15px !important;
        min-height: 15px !important;
        display: block !important;
        color: currentColor !important;
        fill: none !important;
        stroke: currentColor !important;
    }

    .message.bot {
        background: #edf2fb !important;
        border: none !important;
        color: var(--text-color) !important;
        width: min(100%, 1080px) !important;
        max-width: min(100%, 1080px) !important;
        margin-right: auto !important;
        border-radius: 22px !important;
        padding: 18px 22px !important;
    }

    .message.bot > *,
    .message.bot .prose,
    .message.bot .gr-markdown,
    .message.bot [data-testid="markdown"],
    .message.bot .message-content,
    .message.bot .md,
    .message.bot .text,
    .message.bot .component-wrap {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .message.bot .prose,
    .message.bot .gr-markdown,
    .message.bot [data-testid="markdown"],
    .message.bot .message-content,
    .message.bot .md,
    .message.bot .text {
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.72 !important;
    }

    .message.bot p,
    .message.bot .prose p,
    .message.bot .gr-markdown p,
    .message.bot [data-testid="markdown"] p,
    .message.bot .message-content p {
        margin: 0 0 10px !important;
        line-height: 1.72 !important;
    }

    .message.bot p:last-child,
    .message.bot .prose p:last-child,
    .message.bot .gr-markdown p:last-child,
    .message.bot [data-testid="markdown"] p:last-child,
    .message.bot .message-content p:last-child {
        margin-bottom: 0 !important;
    }

    .message.bot ul,
    .message.bot ol,
    .message.bot .prose ul,
    .message.bot .prose ol,
    .message.bot .gr-markdown ul,
    .message.bot .gr-markdown ol,
    .message.bot [data-testid="markdown"] ul,
    .message.bot [data-testid="markdown"] ol {
        margin: 6px 0 12px 1.2em !important;
        padding: 0 !important;
    }

    .message.bot li,
    .message.bot .prose li,
    .message.bot .gr-markdown li,
    .message.bot [data-testid="markdown"] li {
        margin: 0 0 6px !important;
        line-height: 1.7 !important;
    }

    .message.bot h1,
    .message.bot h2,
    .message.bot h3,
    .message.bot h4,
    .message.bot .prose h1,
    .message.bot .prose h2,
    .message.bot .prose h3,
    .message.bot .prose h4,
    .message.bot .gr-markdown h1,
    .message.bot .gr-markdown h2,
    .message.bot .gr-markdown h3,
    .message.bot .gr-markdown h4,
    .message.bot [data-testid="markdown"] h1,
    .message.bot [data-testid="markdown"] h2,
    .message.bot [data-testid="markdown"] h3,
    .message.bot [data-testid="markdown"] h4 {
        margin: 14px 0 8px !important;
        line-height: 1.4 !important;
    }

    .message.bot hr,
    .message.bot .prose hr,
    .message.bot .gr-markdown hr,
    .message.bot [data-testid="markdown"] hr {
        margin: 14px 0 !important;
        border: none !important;
        border-top: 1px solid rgba(148, 163, 184, 0.22) !important;
    }

    .chatbot .message-row {
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 12px !important;
    }

    .chatbot .message-row:has(.message.user) {
        justify-content: flex-end !important;
    }

    .chatbot .message-row:has(.message.bot) {
        justify-content: flex-start !important;
    }

    .message blockquote,
    .message .prose blockquote,
    .message .gr-markdown blockquote,
    .message [data-testid="markdown"] blockquote {
        border-left: none !important;
        border-inline-start: none !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .message blockquote p,
    .message .prose p,
    .message .gr-markdown p,
    .message [data-testid="markdown"] p {
        margin: 0 !important;
    }

    .message.user p,
    .message.user .prose p,
    .message.user .gr-markdown p,
    .message.user [data-testid="markdown"] p,
    .message.user .message-content p {
        display: inline !important;
        white-space: normal !important;
    }

    .message.user br {
        display: none !important;
    }

    .message.user .prose,
    .message.user .prose *,
    .message.user .gr-markdown,
    .message.user .gr-markdown *,
    .message.user [data-testid="markdown"],
    .message.user [data-testid="markdown"] *,
    .message.user .message-content,
    .message.user .message-content *,
    .message.user .md,
    .message.user .md * {
        background-image: none !important;
    }

    .message blockquote::before,
    .message blockquote::after,
    .message .prose blockquote::before,
    .message .prose blockquote::after,
    .message .gr-markdown blockquote::before,
    .message .gr-markdown blockquote::after,
    .message [data-testid="markdown"] blockquote::before,
    .message [data-testid="markdown"] blockquote::after {
        display: none !important;
        content: none !important;
    }

    .message.bot pre,
    .message.bot code {
        background: #edf2ff !important;
        border-radius: 10px !important;
    }

    .chatbot .placeholder,
    .chatbot [class*="placeholder"] {
        color: var(--text-muted) !important;
    }

    .message[data-testid="chatbot-message-status"],
    .message[data-node="status"] {
        display: none !important;
    }

    textarea[placeholder*="输入"],
    textarea[placeholder*="提问"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    form:has(textarea[placeholder*="输入"]),
    form:has(textarea[placeholder*="提问"]) {
        display: flex !important;
        gap: 12px !important;
        padding: 10px 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        border-radius: 18px !important;
        background: rgba(255, 255, 255, 0.84) !important;
    }

    form:has(textarea[placeholder*="输入"]) button,
    form:has(textarea[placeholder*="提问"]) button {
        background: transparent !important;
        padding: 8px !important;
    }

    form:has(textarea[placeholder*="输入"]) button:hover,
    form:has(textarea[placeholder*="提问"]) button:hover {
        background: rgba(49, 88, 214, 0.08) !important;
    }

    .info,
    .gradio-container .description,
    .gradio-container .secondary-text,
    .gradio-container .hint {
        color: var(--text-muted) !important;
    }

    .gr-markdown,
    .gr-markdown *,
    .prose,
    .prose * ,
    p,
    label {
        color: var(--text-color) !important;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 900px) {
        .gradio-container {
            width: min(100vw - 18px, 1000px) !important;
            padding-top: 14px !important;
        }

        .app-hero {
            padding: 22px 20px 20px;
            border-radius: 24px;
        }

        .app-hero h1 {
            font-size: 28px !important;
        }

        .workspace-grid,
        .content-grid {
            flex-direction: column !important;
        }

        .panel {
            padding: 18px !important;
            border-radius: 20px !important;
        }

        .panel-head h2 {
            font-size: 24px !important;
        }

        #file-list-box {
            min-height: 220px !important;
        }
    }
"""
