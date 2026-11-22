// Package planner builds localized byte-range edit plans that make Claude
// Messages request bodies acceptable to Kimi's Anthropic-compatible endpoint:
// nested ToolSearch tool_reference blocks become text, and bodies above the
// byte trigger are trimmed as a growing prefix of one fixed oldest-first
// order, so the trimmed form stabilizes across turns instead of churning.
package planner

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

// PlanOptions pins the trimming policy. Zero budget knobs fall back to
// DefaultPlanOptions; zero protection knobs fall back per field.
type PlanOptions struct {
	// TriggerBytes is the raw-body size above which duplicate images are
	// replaced and trimming becomes possible; FitsKimi requires the final
	// estimate at or below this value.
	TriggerBytes int
	// LandingBytes is the size the quantized budget aims toward.
	LandingBytes int
	// QuantumBytes rounds required savings up so cache-invalidating trims
	// arrive in spaced batches instead of one edit per turn.
	QuantumBytes         int
	ProtectedMessages    int
	ProtectedImages      int
	LargeToolResultBytes int
	ToolResultHeadBytes  int
	ToolResultTailBytes  int
}

// DefaultPlanOptions returns the pinned defaults: trigger = Kimi's hard
// limit minus a 128 KiB margin for executor additions, landing ~25% below
// trigger, quantum 256 KiB.
func DefaultPlanOptions() PlanOptions {
	return PlanOptions{
		TriggerBytes:         1_966_080,
		LandingBytes:         1_500_000,
		QuantumBytes:         262_144,
		ProtectedMessages:    24,
		ProtectedImages:      2,
		LargeToolResultBytes: 16_384,
		ToolResultHeadBytes:  4_096,
		ToolResultTailBytes:  4_096,
	}
}

func (o PlanOptions) normalized() PlanOptions {
	d := DefaultPlanOptions()
	if o.TriggerBytes == 0 {
		o.TriggerBytes = d.TriggerBytes
	}
	if o.LandingBytes == 0 {
		o.LandingBytes = d.LandingBytes
	}
	if o.QuantumBytes == 0 {
		o.QuantumBytes = d.QuantumBytes
	}
	if o.ProtectedMessages <= 0 {
		o.ProtectedMessages = d.ProtectedMessages
	}
	if o.ProtectedImages < 0 {
		o.ProtectedImages = d.ProtectedImages
	}
	if o.LargeToolResultBytes <= 0 {
		o.LargeToolResultBytes = d.LargeToolResultBytes
	}
	if o.ToolResultHeadBytes < 0 {
		o.ToolResultHeadBytes = d.ToolResultHeadBytes
	}
	if o.ToolResultTailBytes < 0 {
		o.ToolResultTailBytes = d.ToolResultTailBytes
	}
	return o
}

func (o PlanOptions) validateBudgets() error {
	if o.TriggerBytes < 0 || o.LandingBytes < 0 || o.QuantumBytes < 0 {
		return fmt.Errorf("budget knobs must be >= 0 (zero = default)")
	}
	if o.TriggerBytes <= o.LandingBytes {
		return fmt.Errorf("trigger_bytes (%d) must be greater than landing_bytes (%d)", o.TriggerBytes, o.LandingBytes)
	}
	if o.QuantumBytes <= 0 {
		return fmt.Errorf("quantum_bytes must be > 0")
	}
	return nil
}

// RequestPlan is the cached output of BuildKimiRequestPlan: a source hash, a
// fit verdict, a post-edit size estimate, and disjoint localized edits.
type RequestPlan struct {
	SourceSHA256   [32]byte
	FitsKimi       bool
	EstimatedBytes int
	Edits          []pluginapi.RequestBodyEdit
	Stats          PlanStats
}

// PlanStats records what each trimming stage changed, for benchmarks and logs.
type PlanStats struct {
	ToolReferences     int
	DuplicateImages    int
	OldImages          int
	TrimmedToolResults int
	SavedBytes         int
}

// minTrimSavings gates optional trims: an edit saving less than this still
// invalidates the provider's cached prefix, so it must not happen.
const minTrimSavings = 4096

var (
	toolRefNeedle    = []byte("tool_reference")
	jsonEscapeNeedle = []byte(`\u`)
)

// MayContainToolReference reports whether body warrants a real parse: the
// literal type name, or any JSON Unicode escape (which may hide one).
func MayContainToolReference(body []byte) bool {
	return bytes.Contains(body, toolRefNeedle) || bytes.Contains(body, jsonEscapeNeedle)
}

type editKind int

const (
	kindToolRef editKind = iota
	kindDupImage
	kindOldImage
	kindToolResult
)

// planEdit is one byte-range replacement under consideration. Duplicate
// markers additionally carry their group's identity and the anchor span they
// reference so the anchor can be repaired if a later trim removes it.
type planEdit struct {
	start, end     int
	repl           string
	kind           editKind
	identity       imageIdentity
	anchorStart    int
	anchorEnd      int
}

type collected struct {
	toolRefs    []*jnode
	images      []*jnode
	toolResults []*jnode
	msgIndex    int
}

// BuildKimiRequestPlan analyzes a Claude Messages body and returns localized
// edits. Stage 1 (tool_reference -> text) always runs. Above TriggerBytes,
// stage 2 replaces every eligible duplicate image unconditionally; if the
// remaining savings are still short of the quantized budget, one merged
// oldest-first list of old images and large tool_result texts is consumed as
// a prefix until the requirement is met. Only the final estimated size at or
// below TriggerBytes sets FitsKimi.
func BuildKimiRequestPlan(body []byte, options PlanOptions) (RequestPlan, error) {
	opts := options.normalized()
	if err := opts.validateBudgets(); err != nil {
		return RequestPlan{}, err
	}
	if len(body) <= opts.TriggerBytes && !MayContainToolReference(body) {
		return RequestPlan{SourceSHA256: sha256.Sum256(body), EstimatedBytes: len(body), FitsKimi: true}, nil
	}
	root, err := parseJSON(body)
	if err != nil {
		return RequestPlan{}, fmt.Errorf("parse request body: %w", err)
	}

	var cols []*collected
	protectedFrom := 0
	if msgs := root.field("messages"); msgs != nil && msgs.kind == kindArray {
		protectedFrom = len(msgs.elems) - opts.ProtectedMessages
		if protectedFrom < 0 {
			protectedFrom = 0
		}
		if active := activeTurnStart(body, msgs.elems); active >= 0 && active < protectedFrom {
			protectedFrom = active
		}
		for i, m := range msgs.elems {
			c := &collected{msgIndex: i}
			collectMessage(body, m, c)
			cols = append(cols, c)
		}
	} else {
		// No messages array: normalize tool_references globally; nothing else
		// is safe to trim and message-level protection does not apply.
		c := &collected{msgIndex: -1}
		walkCollect(body, root, c)
		cols = append(cols, c)
	}

	edits := []planEdit{}
	estimate := len(body)
	accept := func(e planEdit) {
		edits = append(edits, e)
		estimate += len(e.repl) - (e.end - e.start)
	}
	plan := RequestPlan{SourceSHA256: sha256.Sum256(body)}

	// Stage 1: tool_reference normalization — required for validity, not size.
	for _, c := range cols {
		for _, tr := range c.toolRefs {
			name := ""
			if tn := tr.field("tool_name"); tn != nil && tn.kind == kindString {
				if s, err := decodeString(body[tn.start:tn.end]); err == nil {
					name = s
				}
			}
			accept(planEdit{start: tr.start, end: tr.end, repl: textBlockJSON("[tool_reference: " + name + "]"), kind: kindToolRef})
			plan.Stats.ToolReferences++
		}
	}

	type anchorSpan struct{ start, end int }
	anchors := map[imageIdentity]anchorSpan{}
	dupEditIdx := []int{} // edits indexes that are duplicate markers

	// Stage 2: duplicate images, only under byte pressure. Every eligible
	// repeat is replaced: the information survives in the kept anchor.
	if len(body) > opts.TriggerBytes {
		totalImages := countImages(cols)
		imgOrdinal := 0
		for _, c := range cols {
			for _, img := range c.images {
				imgOrdinal++
				protected := (c.msgIndex >= 0 && c.msgIndex >= protectedFrom) ||
					imgOrdinal > totalImages-opts.ProtectedImages
				if protected {
					continue
				}
				id, ok := identifyImage(body, img)
				if !ok {
					continue
				}
				if a, dup := anchors[id]; dup {
					repl := duplicateImageOmission(id, true)
					if img.end-img.start-len(repl) < minTrimSavings {
						continue
					}
					dupEditIdx = append(dupEditIdx, len(edits))
					accept(planEdit{start: img.start, end: img.end, repl: repl, kind: kindDupImage,
						identity: id, anchorStart: a.start, anchorEnd: a.end})
					plan.Stats.DuplicateImages++
					continue
				}
				anchors[id] = anchorSpan{img.start, img.end}
			}
		}
	}

	// Stage-1 growth (replacements longer than the original spans) counts as
	// negative savings: the prefix loop must earn it back.
	saved := len(body) - estimate
	required := quantizedSavings(len(body), opts.LandingBytes, opts.QuantumBytes)

	// Stages 3+4 merged: one oldest-first prefix over old images and large
	// tool_result texts until the quantized requirement is met.
	if len(body) > opts.TriggerBytes && saved < required {
		cands, err := collectTrimCandidates(body, cols, protectedFrom, opts, edits)
		if err != nil {
			return RequestPlan{}, err
		}
		for _, cand := range cands {
			marginal := cand.edit.end - cand.edit.start - len(cand.edit.repl)
			anchorSavings := 0
			if cand.edit.kind == kindOldImage {
				// Removing an anchor also shortens every marker that points at it.
				for _, idx := range dupEditIdx {
					a := anchors[edits[idx].identity]
					if a.start == cand.edit.start && a.end == cand.edit.end {
						anchorSavings += len(edits[idx].repl) - len(duplicateImageOmission(edits[idx].identity, false))
					}
				}
			}
			// The gate applies to the edit's own net savings; anchor-fixup
			// shortening only counts toward prefix selection.
			if marginal < minTrimSavings {
				continue
			}
			accept(cand.edit)
			if cand.edit.kind == kindOldImage {
				plan.Stats.OldImages++
			} else {
				plan.Stats.TrimmedToolResults++
			}
			saved += marginal + anchorSavings
			if saved >= required {
				break
			}
		}
	}

	fixDuplicateAnchors(edits)
	final, err := finalizePlan(body, opts.TriggerBytes, edits, plan.Stats)
	if err != nil {
		return RequestPlan{}, err
	}
	return final, nil
}

// collectMessage gathers typed content blocks of one message, plus the typed
// blocks directly inside a tool_result's content array (where ToolSearch
// places tool_reference blocks).
func collectMessage(body []byte, message *jnode, c *collected) {
	content := message.field("content")
	if content == nil || content.kind != kindArray {
		return
	}
	for _, block := range content.elems {
		collectContentBlock(body, block, c)
	}
}

func collectContentBlock(body []byte, block *jnode, c *collected) {
	if block == nil || block.kind != kindObject {
		return
	}
	switch {
	case block.fieldIs(body, "type", "tool_reference"):
		c.toolRefs = append(c.toolRefs, block)
	case block.fieldIs(body, "type", "image"):
		c.images = append(c.images, block)
	case block.fieldIs(body, "type", "tool_result"):
		c.toolResults = append(c.toolResults, block)
		if content := block.field("content"); content != nil && content.kind == kindArray {
			for _, sub := range content.elems {
				collectContentBlock(body, sub, c)
			}
		}
	}
}

// walkCollect is the no-messages fallback for stage 1 only: it finds
// tool_reference blocks anywhere but must not feed trimming stages.
func walkCollect(body []byte, n *jnode, c *collected) {
	if n == nil {
		return
	}
	if n.kind == kindObject {
		if n.fieldIs(body, "type", "tool_reference") {
			c.toolRefs = append(c.toolRefs, n)
		}
		for _, m := range n.members {
			walkCollect(body, m.value, c)
		}
		return
	}
	if n.kind == kindArray {
		for _, el := range n.elems {
			walkCollect(body, el, c)
		}
	}
}

// activeTurnStart returns the index of the last real user message (one whose
// content is not made exclusively of tool_result blocks), or -1.
func activeTurnStart(body []byte, messages []*jnode) int {
	last := -1
	for i, m := range messages {
		if isRealUserMessage(body, m) {
			last = i
		}
	}
	return last
}

func isRealUserMessage(body []byte, message *jnode) bool {
	if !message.fieldIs(body, "role", "user") {
		return false
	}
	content := message.field("content")
	if content == nil || content.kind != kindArray || len(content.elems) == 0 {
		return true
	}
	for _, el := range content.elems {
		if el.kind != kindObject || !el.fieldIs(body, "type", "tool_result") {
			return true
		}
	}
	return false
}

// imageIdentity groups byte-identical images: SHA-256 over decoded image
// bytes plus media type. Undecodable payloads hash in a distinct fallback
// domain so they never match a decoded image.
type imageIdentity struct {
	digest   [32]byte
	fallback bool
}

func identifyImage(body []byte, img *jnode) (imageIdentity, bool) {
	src := img.field("source")
	if src == nil {
		return imageIdentity{}, false
	}
	data := src.field("data")
	if data == nil || data.kind != kindString {
		return imageIdentity{}, false
	}
	mt := src.field("media_type")
	if mt == nil || mt.kind != kindString {
		return imageIdentity{}, false
	}
	rawData, err := decodeString(body[data.start:data.end])
	if err != nil {
		return imageIdentity{}, false
	}
	media, err := decodeString(body[mt.start:mt.end])
	if err != nil {
		return imageIdentity{}, false
	}
	compact := stripJSONWhitespace(rawData)
	var decoded []byte
	for _, enc := range []*base64.Encoding{
		base64.StdEncoding, base64.URLEncoding, base64.RawStdEncoding, base64.RawURLEncoding,
	} {
		if b, err := enc.DecodeString(compact); err == nil {
			decoded = b
			break
		}
	}
	if decoded != nil {
		return imageIdentity{digest: hashImageParts(decoded, 0, media)}, true
	}
	return imageIdentity{digest: hashImageParts([]byte(rawData), 1, media), fallback: true}, true
}

func hashImageParts(payload []byte, domain byte, media string) [32]byte {
	h := sha256.New()
	h.Write([]byte{domain})
	h.Write(payload)
	h.Write([]byte{0})
	h.Write([]byte(media))
	var sum [32]byte
	copy(sum[:], h.Sum(nil))
	return sum
}

func stripJSONWhitespace(s string) string {
	if !strings.ContainsAny(s, " \t\n\r") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\n', '\r':
		default:
			b.WriteByte(s[i])
		}
	}
	return b.String()
}

func duplicateImageOmission(id imageIdentity, anchorSurvives bool) string {
	if !anchorSurvives {
		return textBlockJSON("[duplicate image omitted]")
	}
	return textBlockJSON(fmt.Sprintf("[duplicate of image %x omitted]", id.digest[:4]))
}

func imageOmission(body []byte, img *jnode) string {
	media := "image"
	if src := img.field("source"); src != nil {
		if mtv := src.field("media_type"); mtv != nil && mtv.kind == kindString {
			if s, err := decodeString(body[mtv.start:mtv.end]); err == nil && s != "" {
				media = s
			}
		}
	}
	return textBlockJSON(fmt.Sprintf("[image (%s, ~%d bytes) omitted to fit request budget]", media, img.end-img.start))
}

// trimCandidate pairs an edit with its message index for age ordering.
type trimCandidate struct {
	msgIndex int
	edit     planEdit
}

// collectTrimCandidates builds the merged oldest-first trim list: remaining
// eligible images (message fence and newest-image fence) and large
// tool_result texts, skipping spans already edited.
func collectTrimCandidates(body []byte, cols []*collected, protectedFrom int, opts PlanOptions, accepted []planEdit) ([]trimCandidate, error) {
	out := []trimCandidate{}
	totalImages := countImages(cols)
	imgOrdinal := 0
	for _, c := range cols {
		if c.msgIndex < 0 {
			continue
		}
		msgProtected := c.msgIndex >= protectedFrom
		for _, img := range c.images {
			imgOrdinal++
			protected := msgProtected || imgOrdinal > totalImages-opts.ProtectedImages
			if protected || overlapsEdits(img.start, img.end, accepted) {
				continue
			}
			id, _ := identifyImage(body, img)
			out = append(out, trimCandidate{c.msgIndex, planEdit{
				start: img.start, end: img.end, repl: imageOmission(body, img), kind: kindOldImage, identity: id,
			}})
		}
		if msgProtected {
			continue
		}
		for _, tr := range c.toolResults {
			for _, sn := range toolResultTextNodes(body, tr) {
				rawLen := sn.strEnd - sn.strStart
				if rawLen <= opts.LargeToolResultBytes || overlapsEdits(sn.start, sn.end, accepted) {
					continue
				}
				decoded, err := decodeString(body[sn.start:sn.end])
				if err != nil {
					continue
				}
				head := headRunes(decoded, opts.ToolResultHeadBytes)
				tail := tailRunes(decoded, opts.ToolResultTailBytes)
				omitted := len(decoded) - len(head) - len(tail)
				if omitted <= 0 {
					continue
				}
				newText := head + "\n…[" + itoa(omitted) + " bytes omitted for length]…\n" + tail
				marshaled, err := json.Marshal(newText)
				if err != nil {
					continue
				}
				out = append(out, trimCandidate{c.msgIndex, planEdit{
					start: sn.start, end: sn.end, repl: string(marshaled), kind: kindToolResult,
				}})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].msgIndex != out[j].msgIndex {
			return out[i].msgIndex < out[j].msgIndex
		}
		return out[i].edit.start < out[j].edit.start
	})
	for i := 1; i < len(out); i++ {
		if out[i].edit.start < out[i-1].edit.end {
			return nil, fmt.Errorf("overlapping trim candidates at %d", out[i].edit.start)
		}
	}
	return out, nil
}

func overlapsEdits(start, end int, accepted []planEdit) bool {
	for _, e := range accepted {
		if e.start < end && start < e.end {
			return true
		}
	}
	return false
}

// quantizedSavings rounds the raw-to-landing shortfall up to a quantum
// multiple; all arithmetic in uint64 to survive extreme configured values.
func quantizedSavings(raw, landing, quantum int) int {
	if raw <= landing {
		return 0
	}
	diff := uint64(raw - landing)
	q := uint64(quantum)
	return int((diff + q - 1) / q * q)
}

// fixDuplicateAnchors downgrades duplicate markers whose anchor span was
// removed by another accepted edit to a static omission marker.
func fixDuplicateAnchors(edits []planEdit) {
	for i := range edits {
		e := &edits[i]
		if e.kind != kindDupImage {
			continue
		}
		for j := range edits {
			if i == j {
				continue
			}
			if edits[j].start < e.anchorEnd && e.anchorStart < edits[j].end {
				e.repl = duplicateImageOmission(e.identity, false)
				break
			}
		}
	}
}

// finalizePlan re-sorts edits by source position, validates disjointness,
// and recomputes the estimate once from the final replacement texts.
func finalizePlan(body []byte, trigger int, edits []planEdit, stats PlanStats) (RequestPlan, error) {
	sort.Slice(edits, func(i, j int) bool { return edits[i].start < edits[j].start })
	for i := 1; i < len(edits); i++ {
		if edits[i].start < edits[i-1].end {
			return RequestPlan{}, fmt.Errorf("overlapping edits at %d", edits[i].start)
		}
	}
	plan := RequestPlan{SourceSHA256: sha256.Sum256(body), Stats: stats}
	plan.Edits = make([]pluginapi.RequestBodyEdit, 0, len(edits))
	estimate := len(body)
	for _, e := range edits {
		plan.Edits = append(plan.Edits, pluginapi.RequestBodyEdit{Start: e.start, End: e.end, Replacement: e.repl})
		estimate += len(e.repl) - (e.end - e.start)
	}
	plan.EstimatedBytes = estimate
	plan.FitsKimi = estimate <= trigger
	plan.Stats.SavedBytes = len(body) - estimate
	return plan, nil
}

func countImages(cols []*collected) int {
	total := 0
	for _, c := range cols {
		total += len(c.images)
	}
	return total
}

// toolResultTextNodes returns the string nodes carrying a tool_result's text:
// a direct string content value, plain string elements, or text-block fields.
func toolResultTextNodes(body []byte, tr *jnode) []*jnode {
	out := []*jnode{}
	content := tr.field("content")
	if content == nil {
		return out
	}
	if content.kind == kindString {
		return append(out, content)
	}
	if content.kind != kindArray {
		return out
	}
	for _, el := range content.elems {
		switch {
		case el.kind == kindString:
			out = append(out, el)
		case el.fieldIs(body, "type", "text"):
			if t := el.field("text"); t != nil && t.kind == kindString {
				out = append(out, t)
			}
		}
	}
	return out
}

func textBlockJSON(text string) string {
	marshaled, _ := json.Marshal(text)
	return `{"type":"text","text":` + string(marshaled) + `}`
}

// headRunes keeps the first n bytes of s, backing off to a rune boundary.
func headRunes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

// tailRunes keeps the last n bytes of s, advancing to a rune boundary.
func tailRunes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := len(s) - n
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return s[cut:]
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
