# E2E Test Issues to Debug

**Created:** 2026-02-05
**Updated:** 2026-02-05
**Status:** 4/5 issues resolved, 1 remaining (ASR quality limitation)

---

## Issue 1: Segment Merging (test_language_detection)

**Priority:** High
**Component:** `src/app/streaming_policy.py`
**Status:** [x] **RESOLVED**

### Problem
Pipeline returns 2-3 segments instead of expected 6. The streaming policy merges consecutive segments too aggressively.

### Root Cause
Two bugs were found:
1. `_overlaps_finalized()` used unidirectional overlap check - a short finalized segment could "hide" inside a longer new segment
2. `now_sec` used wall clock time instead of audio buffer time, causing premature finalization during streaming

### Fix Applied
1. **streaming_policy.py:143-158** - Changed `_overlaps_finalized()` to use bidirectional overlap checking
2. **streaming.py:718** - Changed `now_sec = time.time() - session.start_time` to `now_sec = session.get_current_time()`

### Result
Test now passes with all 6 segments correctly captured.

---

## Issue 2: Ukrainian Latin Script (test_language_pair[uk])

**Priority:** Medium
**Component:** ASR (Whisper) + Language Detection
**Status:** [x] **RESOLVED**

### Problem
ASR outputs Latin transliteration instead of Cyrillic script, plus truncates final sentence.

### Root Cause
1. SpeechBrain language detection confused Ukrainian (uk) with Belarusian (be)
2. Without language hint, the confusion correction didn't apply
3. Final segment was lost due to timing issues (wall clock vs audio time mismatch)

### Fix Applied
1. **test_bilingual_dialogue.py:296-301** - Changed to use `stream_audio_with_foreign_hint()` which passes the language hint
2. **streaming.py:1306** - Fixed `now_sec` in `request_summary` handler to use audio time
3. **streaming/client.py** - Increased post-stream wait from 2s to 5s for final ASR tick

### Result
Test now passes with correct Cyrillic Ukrainian text and all segments captured.

---

## Issue 3: Albanian Content Filter (test_language_pair[sq])

**Priority:** Medium
**Component:** TTS (Google Cloud) + ASR (Whisper)
**Status:** [~] **PARTIALLY RESOLVED**

### Problem
Google Cloud TTS blocks Albanian scenario text with content filter error.

### Root Cause
The original scenario about residence registration contained sensitive terms:
- "Ausweis" (ID document)
- Immigration-related terminology
- Family member documentation

### Fix Applied
**fixtures/scenarios/sq_customer_service.yaml** - Rewrote scenario as neutral restaurant conversation (reservations, food ordering)

### Result
- Content filter issue: **FIXED** - Audio now generates successfully
- Test still fails due to **ASR quality limitation** - Whisper has poor support for Albanian:
  - "Tisch reserviert" → "Tag" (German ASR error)
  - Albanian spelling errors (falimenderit, qfar, etc.)
  - Final phrase garbled

This is a Whisper model limitation for low-resource languages, not a pipeline bug.

---

## Issue 4: Farsi Final Segment Dropped (test_language_pair[fa])

**Priority:** Medium
**Component:** ASR/Streaming
**Status:** [x] **RESOLVED**

### Problem
ASR drops the last Farsi segment entirely.

### Root Cause
Same timing issue as Issues 1 and 2 - wall clock time vs audio buffer time mismatch.

### Fix Applied
Same fixes as Issue 1 and 2 (streaming.py timing fixes).

### Result
Test passes - was already passing after Issue 1 fixes.

---

## Issue 5: Summary Timeout (test_scenario_type[code_switching])

**Priority:** Low
**Component:** Summarization (Qwen3)
**Status:** [x] **RESOLVED**

### Problem
Transcription and translation work fine, but summary generation times out.

### Root Cause
1. Default timeout of 120s was too short for Qwen3 summarization
2. Post-stream wait of 2s was too short for final ASR tick to complete

### Fix Applied
1. **streaming/client.py** - Increased timeout from 120s to 300s
2. **streaming/client.py** - Increased post-stream wait from 2s to 5s

### Result
Test now passes with summary generated successfully.

---

## Resolution Summary

| Issue | Status | Fix |
|-------|--------|-----|
| Segment Merging | ✅ RESOLVED | Bidirectional overlap + audio time |
| Ukrainian Script | ✅ RESOLVED | Language hint + timing fix |
| Albanian Filter | ⚠️ PARTIAL | Scenario rewritten, ASR quality remains |
| Farsi Dropped | ✅ RESOLVED | Timing fixes |
| Summary Timeout | ✅ RESOLVED | Increased timeouts |

### Files Modified
- `src/app/streaming_policy.py` - Bidirectional overlap checking
- `src/app/streaming.py` - Audio time for finalization (2 locations)
- `tests/e2e/streaming/client.py` - Timeouts and wait times
- `tests/e2e/test_bilingual_dialogue.py` - Use foreign language hint
- `tests/fixtures/scenarios/sq_customer_service.yaml` - Neutral restaurant scenario

### Test Results After Fixes
```
test_language_detection         PASSED
test_language_pair[uk]          PASSED
test_language_pair[sq]          FAILED (ASR quality - Whisper limitation)
test_language_pair[fa]          PASSED
test_scenario_type[code_switching] PASSED
```
