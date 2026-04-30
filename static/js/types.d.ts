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
