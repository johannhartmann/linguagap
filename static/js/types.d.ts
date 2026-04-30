/**
 * Ambient type declarations for the LinguaGap frontend.
 *
 * The static pages each load several plain `<script>` tags that share a
 * window-level namespace. This file gives TypeScript visibility into the
 * globals those scripts publish so the per-page modules can reference them
 * without `any` everywhere.
 */

/** Bilingual transcript exporter — see static/js/transcript_export.js. */
declare const TranscriptExport: {
    /**
     * @param segments  Server-provided segments (each must have `final`, `src`,
     *                  `src_lang`, `abs_start`, `translations`).
     * @param foreignLang  BCP-47 short code of the non-German language.
     * @param source  Tag identifying the caller (e.g. "host", "viewer").
     * @param langNames  Optional mapping from code to human-readable label.
     */
    buildHtml(
        segments: ReadonlyArray<unknown>,
        foreignLang: string | null,
        source: string,
        langNames?: Record<string, string> | null
    ): string;
    filename(foreignLang: string | null): string;
    download(opts: {
        segments: ReadonlyArray<unknown>;
        foreignLang: string | null;
        source: string;
        langNames?: Record<string, string> | null;
    }): void;
};

/** QR code library — see static/js/vendor/qrcode.js (kazuhikoarase, MIT).
 *
 * The library exposes many internal methods (getModuleCount, isDark, etc.)
 * that vary between rendering paths; we leave the instance shape open with
 * an index signature so the consumer can use whichever helpers it needs.
 */
declare const qrcode: ((
    typeNumber: number,
    errorCorrectionLevel: 'L' | 'M' | 'Q' | 'H'
) => {
    addData(data: string): void;
    make(): void;
    createSvgTag(opts: { cellSize?: number; margin?: number }): string;
    [key: string]: any;
}) & { [key: string]: any };

/** Shared auth guard — see static/js/lib/auth_guard.js. */
declare const LinguaGapAuth: {
    requireUser(): Promise<{
        email: string;
        display_name: string;
        logo_url: string;
        is_admin: boolean;
    } | null>;
    wireLogoutButton(buttonId: string, redirectTo?: string): void;
};

/** Shared i18n resolver — see static/js/lib/i18n.js. */
declare const LinguaGapI18n: {
    t(
        maps: ReadonlyArray<Record<string, Record<string, string>>>,
        currentLang: string | null | undefined,
        key: string,
        replacements?: Record<string, string | number>
    ): string;
};

/**
 * Chrome historically exposed extra google-prefixed audio constraints. They
 * still work but aren't part of the standard MediaTrackConstraints typing.
 */
interface MediaTrackConstraintSet {
    googEchoCancellation?: boolean;
    googAutoGainControl?: boolean;
    googNoiseSuppression?: boolean;
    googHighpassFilter?: boolean;
    googTypingNoiseDetection?: boolean;
}
