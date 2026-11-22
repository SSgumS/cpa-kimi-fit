package planner

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

func applyEdits(t *testing.T, body []byte, edits []pluginapi.RequestBodyEdit) []byte {
	t.Helper()
	out := append([]byte(nil), body...)
	for i := len(edits) - 1; i >= 0; i-- {
		e := edits[i]
		if e.Start < 0 || e.End > len(out) || e.Start > e.End {
			t.Fatalf("edit %d out of range: %d..%d (len %d)", i, e.Start, e.End, len(out))
		}
		repl := e.Replacement
		if len(e.ReplacementBytes) > 0 {
			repl = string(e.ReplacementBytes)
		}
		next := make([]byte, 0, len(out)-(e.End-e.Start)+len(repl))
		next = append(next, out[:e.Start]...)
		next = append(next, repl...)
		next = append(next, out[e.End:]...)
		out = next
	}
	return out
}

// mustApplyValid asserts the splice invariants: edits sorted and disjoint,
// valid JSON result, and exact accounting on every plan.
func mustApplyValid(t *testing.T, body []byte, plan RequestPlan) []byte {
	t.Helper()
	for i := 1; i < len(plan.Edits); i++ {
		if plan.Edits[i].Start < plan.Edits[i-1].End {
			t.Fatalf("edits not sorted/disjoint at %d", i)
		}
	}
	out := applyEdits(t, body, plan.Edits)
	if !json.Valid(out) {
		t.Fatalf("edited body is not valid JSON:\n%s", out[:min(400, len(out))])
	}
	if plan.EstimatedBytes != len(out) {
		t.Fatalf("estimate %d != applied size %d", plan.EstimatedBytes, len(out))
	}
	if plan.Stats.SavedBytes != len(body)-len(out) {
		t.Fatalf("SavedBytes %d != %d", plan.Stats.SavedBytes, len(body)-len(out))
	}
	return out
}

// pressureOptions returns options that force the full trim machinery on a
// fixture: trigger just under the body, landing tiny, quantum 1 (no rounding).
func pressureOptions(bodyLen int) PlanOptions {
	o := DefaultPlanOptions()
	o.TriggerBytes = bodyLen - 1
	o.LandingBytes = 64
	o.QuantumBytes = 1
	return o
}

func toolSearchFixture() []byte {
	return []byte(`{"model":"k3.1-pro","messages":[` +
		`{"role":"user","content":[{"type":"text","text":"hi"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ToolSearch","input":{"query":"x"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` +
		`{"type":"tool_reference","tool_name":"WebSearch"},` +
		`{"type":"tool_reference","tool_name":"WebFetch"}]}]},` +
		`{"role":"user","content":[{"type":"text","text":"latest"}]}` +
		`]}`)
}

func imageBlock(b64 string) string {
	return `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64 + `"}}`
}

func imageBlockMedia(b64, media string) string {
	return `{"type":"image","source":{"type":"base64","media_type":"` + media + `","data":"` + b64 + `"}}`
}

// bigB64 returns base64 data of roughly n bytes (4-char units).
func bigB64(n int) string {
	return strings.Repeat("QUJD", n/4+1)
}

// mustImageNode parses body and returns its single image content block node.
func mustImageNode(t *testing.T, body []byte) *jnode {
	t.Helper()
	root, err := parseJSON(body)
	if err != nil {
		t.Fatal(err)
	}
	return root.field("messages").elems[0].field("content").elems[0]
}

// imageFixtureBody wraps one image block in a single-message body.
func imageFixtureBody(block string) []byte {
	return []byte(`{"messages":[{"role":"user","content":[` + block + `]}]}`)
}

// expectedImageHash mirrors hashImageParts for a decoded payload.
func expectedImageHash(decoded []byte, media string) string {
	h := sha256.New()
	h.Write([]byte{0})
	h.Write(decoded)
	h.Write([]byte{0})
	h.Write([]byte(media))
	return fmt.Sprintf("%x", h.Sum(nil)[:4])
}

func TestSmallBodyFits(t *testing.T) {
	body := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	plan, err := BuildKimiRequestPlan(body, DefaultPlanOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !plan.FitsKimi || len(plan.Edits) != 0 {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestToolReferenceNormalization(t *testing.T) {
	body := toolSearchFixture()
	plan, err := BuildKimiRequestPlan(body, DefaultPlanOptions())
	if err != nil {
		t.Fatal(err)
	}
	if plan.Stats.ToolReferences != 2 {
		t.Fatalf("tool refs = %d, want 2", plan.Stats.ToolReferences)
	}
	out := string(mustApplyValid(t, body, plan))
	if strings.Contains(out, `"type":"tool_reference"`) {
		t.Fatalf("tool_reference survives: %s", out)
	}
	if !strings.Contains(out, "WebSearch") || !strings.Contains(out, "WebFetch") {
		t.Fatalf("tool names lost: %s", out)
	}
}

// An escaped tool_reference must still normalize: the literal needle misses
// it, so the \u fast-path bypass must catch it. The type name below is a
// real JSON unicode escape ("_" = _), not the literal string.
func TestEscapedToolReferenceBypassesFastPath(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` +
		`{"type":"tool\u005freference","tool_name":"WebSearch"}]}]}]}`)
	if strings.Contains(string(body), "tool_reference") {
		t.Fatal("fixture must not contain the literal type name")
	}
	if !MayContainToolReference(body) {
		t.Fatal("escaped tool_reference not detected by fast-path guard")
	}
	plan, err := BuildKimiRequestPlan(body, DefaultPlanOptions())
	if err != nil {
		t.Fatal(err)
	}
	if plan.Stats.ToolReferences != 1 {
		t.Fatalf("tool refs = %d, want 1", plan.Stats.ToolReferences)
	}
	out := string(mustApplyValid(t, body, plan))
	if strings.Contains(out, `"type":"tool_reference"`) {
		t.Fatalf("escaped tool_reference survives: %s", out)
	}
	if !strings.Contains(out, "[tool_reference: WebSearch]") {
		t.Fatalf("escaped tool_reference not normalized: %s", out)
	}
}

func TestDuplicateImageDedup(t *testing.T) {
	b64 := bigB64(4400)
	dup := []byte(`{"messages":[` +
		`{"role":"user","content":[` + imageBlock(b64) + `]},` +
		`{"role":"user","content":[` + imageBlock(b64) + `]},` +
		`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
		`]}`)
	opts := pressureOptions(len(dup))
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	// Dedup alone (~4.4 KB) must satisfy the budget so the anchor survives.
	opts.LandingBytes = len(dup) - 4000
	opts.QuantumBytes = 1
	plan, err := BuildKimiRequestPlan(dup, opts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Stats.DuplicateImages != 1 {
		t.Fatalf("dup images = %d, want 1", plan.Stats.DuplicateImages)
	}
	decoded, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		t.Fatal(err)
	}
	want := fmt.Sprintf("[duplicate of image %s omitted]", expectedImageHash(decoded, "image/png"))
	out := string(mustApplyValid(t, dup, plan))
	if strings.Count(out, b64) != 1 {
		t.Fatal("first image not kept or second not replaced")
	}
	if !strings.Contains(out, want) {
		t.Fatalf("missing dedup placeholder %q", want)
	}
}

func TestOldImagesTrimmedOldestFirst(t *testing.T) {
	pads := []string{bigB64(4400), bigB64(4500), bigB64(4600), bigB64(4700), bigB64(4800), bigB64(5000)}
	var sb strings.Builder
	sb.WriteString(`{"messages":[`)
	for i, p := range pads {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(`{"role":"user","content":[` + imageBlock(p) + `]}`)
	}
	sb.WriteString(`]}`)
	body := []byte(sb.String())
	opts := pressureOptions(len(body))
	opts.ProtectedImages = 1
	opts.ProtectedMessages = 1
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	if len(out) > opts.TriggerBytes {
		t.Fatalf("result %d bytes > trigger %d", len(out), opts.TriggerBytes)
	}
	// Newest image (ordinal 6, protected) must survive untouched.
	if strings.Count(out, `"type":"image"`) != 1 {
		t.Fatalf("expected exactly the protected image to remain")
	}
	// Placeholders must not carry traversal ordinals.
	if strings.Contains(out, "image #") {
		t.Fatal("ordinal leaked into placeholder")
	}
}

func TestProtectedMessagesUntouched(t *testing.T) {
	big := strings.Repeat("x", 40000)
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"` + big + `"}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"` + big + `"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.LargeToolResultBytes = 1000
	opts.ToolResultHeadBytes = 50
	opts.ToolResultTailBytes = 50
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	if plan.Stats.TrimmedToolResults != 1 {
		t.Fatalf("trimmed results = %d, want 1 (protected msg must not trim)", plan.Stats.TrimmedToolResults)
	}
	if strings.Count(out, `"tool_use_id":"b","content":"`+strings.Repeat("x", 100)) != 1 {
		t.Fatal("protected message content was modified")
	}
}

func TestToolResultHeadTailRuneSafe(t *testing.T) {
	content := strings.Repeat("é", 3000) + strings.Repeat("世", 3000)
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"` + content + `"}]}]},` +
		`{"role":"user","content":[{"type":"text","text":"last"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.LargeToolResultBytes = 1000
	opts.ToolResultHeadBytes = 100
	opts.ToolResultTailBytes = 100
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := mustApplyValid(t, body, plan)
	if plan.Stats.TrimmedToolResults != 1 {
		t.Fatalf("trimmed = %d", plan.Stats.TrimmedToolResults)
	}
	if !strings.Contains(string(out), "bytes omitted for length") {
		t.Fatalf("missing omission marker: %s", out[:300])
	}
	if !strings.HasPrefix(string(out), `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t"`) {
		t.Fatal("tool_result structure altered")
	}
}

func TestMalformedJSONFailsOpen(t *testing.T) {
	body := append([]byte(`{"messages":[{broken`), strings.Repeat("x", 2000)...)
	opts := pressureOptions(len(body))
	if _, err := BuildKimiRequestPlan(body, opts); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestNonFittingWhenProtectedTooBig(t *testing.T) {
	big := strings.Repeat("y", 50000)
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"` + big + `"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.LargeToolResultBytes = 1000
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.FitsKimi {
		t.Fatalf("plan should not fit; stats %+v", plan.Stats)
	}
	if len(plan.Edits) != 0 {
		t.Fatalf("edits applied to protected content")
	}
}

func TestIdempotentSecondPlan(t *testing.T) {
	body := toolSearchFixture()
	plan1, err := BuildKimiRequestPlan(body, DefaultPlanOptions())
	if err != nil {
		t.Fatal(err)
	}
	out := applyEdits(t, body, plan1.Edits)
	plan2, err := BuildKimiRequestPlan(out, DefaultPlanOptions())
	if err != nil {
		t.Fatal(err)
	}
	if plan2.Stats.ToolReferences != 0 {
		t.Fatalf("second plan re-normalized: %+v", plan2.Stats)
	}
}

func TestParseScannerBasics(t *testing.T) {
	root, err := parseJSON([]byte(`{"a":[1,"x\ny",{"b":null}],"c":-1.5e3}`))
	if err != nil {
		t.Fatal(err)
	}
	if root.field("a") == nil || root.field("c") == nil {
		t.Fatal("fields missing")
	}
	arr := root.field("a")
	if len(arr.elems) != 3 {
		t.Fatalf("elems = %d", len(arr.elems))
	}
	if arr.elems[1].kind != kindString {
		t.Fatal("string node expected")
	}
}

func TestScannerStrictness(t *testing.T) {
	bad := map[string][]byte{
		"invalid escape":       []byte(`{"a":"\q"}`),
		"raw control byte":     []byte("{\"a\":\"x\ry\"}"),
		"leading zero":         []byte(`{"a":01}`),
		"double fraction":      []byte(`{"a":1.2.3}`),
		"plus sign":            []byte(`{"a":+1}`),
		"empty exponent":       []byte(`{"a":1e}`),
		"double minus":         []byte(`{"a":--5}`),
		"bare minus":           []byte(`{"a":-}`),
		"dangling escape":      []byte(`{"a":"x\`),
		"bad unicode escape":   []byte(`{"a":"\u12xz"}`),
		"short unicode escape": []byte(`{"a":"\u12"}`),
	}
	for name, body := range bad {
		t.Run(name, func(t *testing.T) {
			if _, err := parseJSON(body); err == nil {
				t.Fatalf("malformed input accepted: %s", body)
			}
		})
	}
	good := map[string][]byte{
		"escapes":    []byte(`{"a":"\\\/\b\f\n\r\t"}`),
		"unicode":    []byte(`{"a":"Aé世"}`),
		"numbers":    []byte(`{"a":-0.5e+10,"b":0,"c":[1,2.5,-3E-2]}`),
		"empty str":  []byte(`{"a":""}`),
		"deep nest":  []byte(`{"a":[{"b":{"c":[null,true,false]}}]}`),
		"zero point": []byte(`{"a":0.25}`),
	}
	for name, body := range good {
		t.Run("valid_"+name, func(t *testing.T) {
			if _, err := parseJSON(body); err != nil {
				t.Fatalf("valid input rejected: %s: %v", body, err)
			}
		})
	}
}

// TestActiveTurnFence: a mixed text+tool_result user message starts the
// active turn; everything from there on stays byte-identical even when the
// trailing 24-message window alone would leave it exposed.
func TestActiveTurnFence(t *testing.T) {
	big := strings.Repeat("x", 20000)
	tr := func(id string) string {
		return `[{"type":"tool_result","tool_use_id":"` + id + `","content":"` + big + `"}]`
	}
	body := []byte(`{"messages":[` +
		`{"role":"user","content":` + tr("old") + `},` + // 0: old result, trimmable
		`{"role":"user","content":[{"type":"text","text":"mixed"},{"type":"tool_result","tool_use_id":"m","content":"` + big + `"}]},` + // 1: real user
		`{"role":"assistant","content":[{"type":"text","text":"` + big + `"}]},` + // 2
		`{"role":"user","content":` + tr("late") + `}` + // 3: tool_result-only user
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.LargeToolResultBytes = 1000
	opts.ToolResultHeadBytes = 50
	opts.ToolResultTailBytes = 50
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	if plan.Stats.TrimmedToolResults != 1 {
		t.Fatalf("trimmed = %d, want only the pre-active-turn result", plan.Stats.TrimmedToolResults)
	}
	if !strings.Contains(out, `"tool_use_id":"m","content":"`+strings.Repeat("x", 200)) {
		t.Fatal("active-turn tool_result was modified")
	}
	if !strings.Contains(out, `"text":"`+strings.Repeat("x", 200)) {
		t.Fatal("assistant message inside active turn was modified")
	}
	if !strings.Contains(out, `"tool_use_id":"late","content":"`+strings.Repeat("x", 200)) {
		t.Fatal("tool_result-only user message inside active turn was modified")
	}
}

func TestActiveTurnFenceCases(t *testing.T) {
	big := strings.Repeat("x", 20000)
	tr := func(id string) string {
		return `[{"type":"tool_result","tool_use_id":"` + id + `","content":"` + big + `"}]`
	}
	cases := []struct {
		name        string
		messages    string
		wantTrimmed int
	}{
		{
			// Escaped tool_result types must not count as real user messages:
			// both escaped tool_result-only users predate the real user and
			// both big results are trimmable. The escapes are real JSON
			// unicode escapes ("_" = _).
			name: "escaped type is not a real user",
			messages: `{"role":"user","content":[{"type":"tool\u005fresult","tool_use_id":"e1","content":"` + big + `"}]},` +
				`{"role":"user","content":[{"type":"tool\u005fresult","tool_use_id":"e2","content":"` + big + `"}]},` +
				`{"role":"user","content":[{"type":"text","text":"real"}]}`,
			wantTrimmed: 2,
		},
		{
			name:        "empty content array is a real user",
			messages:    `{"role":"user","content":` + tr("a") + `},{"role":"user","content":[]}`,
			wantTrimmed: 1,
		},
		{
			name:        "no real user falls back to window",
			messages:    `{"role":"user","content":` + tr("a") + `},{"role":"assistant","content":[{"type":"text","text":"ok"}]}`,
			wantTrimmed: 1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(`{"messages":[` + tc.messages + `]}`)
			opts := pressureOptions(len(body))
			opts.ProtectedMessages = 1
			opts.LargeToolResultBytes = 1000
			opts.ToolResultHeadBytes = 50
			opts.ToolResultTailBytes = 50
			plan, err := BuildKimiRequestPlan(body, opts)
			if err != nil {
				t.Fatal(err)
			}
			mustApplyValid(t, body, plan)
			if plan.Stats.TrimmedToolResults != tc.wantTrimmed {
				t.Fatalf("trimmed = %d, want %d (stats %+v)", plan.Stats.TrimmedToolResults, tc.wantTrimmed, plan.Stats)
			}
		})
	}
}

// TestProtocolSpansUntouched: system, tools, thinking/signatures, and
// block-shaped JSON inside tool_use.input survive; a real old result trims
// without losing its ID; with no messages array nothing is trimmed.
func TestProtocolSpansUntouched(t *testing.T) {
	big := strings.Repeat("z", 30000)
	body := []byte(`{"system":"sys-mark-0123456789",` +
		`"tools":[{"name":"f","input_schema":{"type":"object"}}],` +
		`"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"old","content":"` + big + `"}]},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"sig-aaa-bbb","signature":"sig-mark-1"},{"type":"tool_use","id":"u1","name":"f","input":{"query":"{\"type\":\"image\"}"}}]},` +
		`{"role":"user","content":[{"type":"text","text":"last"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.LargeToolResultBytes = 1000
	opts.ToolResultHeadBytes = 50
	opts.ToolResultTailBytes = 50
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	if !strings.Contains(out, "sys-mark-0123456789") || !strings.Contains(out, "sig-mark-1") {
		t.Fatal("system/thinking spans were modified")
	}
	if !strings.Contains(out, `"input_schema":{"type":"object"}`) {
		t.Fatal("tools array was modified")
	}
	if !strings.Contains(out, `{"query":"{\"type\":\"image\"}"}`) {
		t.Fatal("tool_use input was modified")
	}
	if !strings.Contains(out, `"tool_use_id":"old","content":"`) {
		t.Fatal("old tool_result ID/pairing lost")
	}

	// No messages array: stage 1 may run, but no trimming.
	noMsgs := []byte(`{"data":{"nested":[{"type":"tool_reference","tool_name":"X"}]}}`)
	noMsgsOpts := DefaultPlanOptions()
	noMsgsOpts.TriggerBytes = len(noMsgs) - 1
	noMsgsOpts.LandingBytes = 1
	noMsgsOpts.QuantumBytes = 1
	plan2, err := BuildKimiRequestPlan(noMsgs, noMsgsOpts)
	if err != nil {
		t.Fatal(err)
	}
	if plan2.Stats.OldImages != 0 || plan2.Stats.TrimmedToolResults != 0 || plan2.Stats.DuplicateImages != 0 {
		t.Fatalf("trimming ran without a messages array: %+v", plan2.Stats)
	}
}

func TestImageIdentity(t *testing.T) {
	stdB64 := bigB64(4400)
	urlSafe := strings.NewReplacer("+", "-", "/", "_").Replace(stdB64)
	rawURLSafe := strings.TrimRight(urlSafe, "=")
	// Inject a JSON-escaped newline (whitespace) into the encoded data.
	wsEscaped := stdB64[:40] + `\n` + stdB64[40:]

	idStd, ok := identifyImage(imageFixtureBody(imageBlock(stdB64)), mustImageNode(t, imageFixtureBody(imageBlock(stdB64))))
	if !ok || idStd.fallback {
		t.Fatal("standard encoding not identified as decoded")
	}
	same := map[string]string{
		"url-safe alphabet":  urlSafe,
		"url-safe raw":       rawURLSafe,
		"whitespace escaped": wsEscaped,
	}
	for name, b64 := range same {
		body := imageFixtureBody(imageBlock(b64))
		id, ok := identifyImage(body, mustImageNode(t, body))
		if !ok || id != idStd {
			t.Fatalf("%s: identity mismatch (ok=%v)", name, ok)
		}
	}
	diffBody := imageFixtureBody(imageBlock(stdB64[:len(stdB64)-8] + "QUJERk8"))
	diff, ok := identifyImage(diffBody, mustImageNode(t, diffBody))
	if !ok || diff == idStd {
		t.Fatal("different payload hashed equal")
	}
	otherMediaBody := imageFixtureBody(imageBlockMedia(stdB64, "image/jpeg"))
	otherMedia, ok := identifyImage(otherMediaBody, mustImageNode(t, otherMediaBody))
	if !ok || otherMedia == idStd {
		t.Fatal("media type not part of identity")
	}
	invalid := strings.Repeat("!", 64)
	inv1, ok1 := identifyImage(imageFixtureBody(imageBlock(invalid)), mustImageNode(t, imageFixtureBody(imageBlock(invalid))))
	inv2, ok2 := identifyImage(imageFixtureBody(imageBlock(invalid)), mustImageNode(t, imageFixtureBody(imageBlock(invalid))))
	if !ok1 || !ok2 || !inv1.fallback || inv1 != inv2 {
		t.Fatal("invalid payloads not grouped in fallback domain")
	}
	if inv1 == idStd {
		t.Fatal("fallback domain collides with decoded domain")
	}
}

func TestQuantizedSavings(t *testing.T) {
	if got := quantizedSavings(1000, 1500, 256); got != 0 {
		t.Fatalf("below landing: got %d", got)
	}
	if got := quantizedSavings(1500+256, 1500, 256); got != 256 {
		t.Fatalf("exact quantum: got %d", got)
	}
	if got := quantizedSavings(1500+257, 1500, 256); got != 512 {
		t.Fatalf("one past quantum: got %d", got)
	}
	if got := quantizedSavings(1500+1, 1500, 1<<62); got != 1 << 62 {
		t.Fatalf("huge quantum overflow: got %d", got)
	}
}

func TestOverlapsEdits(t *testing.T) {
	accepted := []planEdit{{start: 10, end: 20}, {start: 30, end: 40}}
	cases := []struct {
		start, end int
		want       bool
	}{
		{0, 10, false}, // touching half-open boundary
		{20, 30, false},
		{15, 25, true}, // child inside
		{5, 15, true},  // parent overlap
		{35, 50, true}, // crosses second
		{50, 60, false},
	}
	for _, tc := range cases {
		if got := overlapsEdits(tc.start, tc.end, accepted); got != tc.want {
			t.Fatalf("overlapsEdits(%d,%d) = %v, want %v", tc.start, tc.end, got, tc.want)
		}
	}
}

// TestQuantizedTrimPrefix: the merged order consumes candidates oldest-first
// across kinds, replanning is bit-identical, and at raw == trigger nothing
// but stage 1 runs.
func TestQuantizedTrimPrefix(t *testing.T) {
	mkResult := func(id string, size int) string {
		return `{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + id + `","content":"` + strings.Repeat("x", size) + `"}]}`
	}
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[` + imageBlock(bigB64(4400)) + `]},` + // older image
		mkResult("r1", 20000) + `,` + // younger large result, before the real user
		`{"role":"user","content":[{"type":"text","text":"current question"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	opts.LandingBytes = len(body) - 4400 - 100 // require roughly the image savings
	opts.QuantumBytes = 8192                   // rounds the requirement up to 8192
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	imgIdx := strings.Index(out, "omitted to fit request budget")
	resIdx := strings.Index(out, "bytes omitted for length")
	if imgIdx < 0 || resIdx < 0 {
		t.Fatalf("both candidates should trim: %s", out[:300])
	}
	if imgIdx > resIdx {
		t.Fatal("age order violated: younger result trimmed before older image")
	}
	plan2, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan, plan2) {
		t.Fatal("planner not deterministic for identical input")
	}
	// At raw == trigger nothing but stage 1 may run.
	dupBody := []byte(`{"messages":[` +
		`{"role":"user","content":[` + imageBlock(bigB64(4400)) + `]},` +
		`{"role":"user","content":[` + imageBlock(bigB64(4400)) + `]}` +
		`]}`)
	atTrigger := DefaultPlanOptions()
	atTrigger.TriggerBytes = len(dupBody)
	atTrigger.LandingBytes = 1000
	atTrigger.QuantumBytes = 1
	plan3, err := BuildKimiRequestPlan(dupBody, atTrigger)
	if err != nil {
		t.Fatal(err)
	}
	if plan3.Stats.DuplicateImages != 0 || plan3.Stats.OldImages != 0 {
		t.Fatalf("trims ran at raw == trigger: %+v", plan3.Stats)
	}
}

// TestTrimSavingsGate: edits saving less than 4096 bytes never happen. The
// image message is followed by one filler so the trailing window leaves it
// eligible.
func TestTrimSavingsGate(t *testing.T) {
	find := func(target int) []byte {
		t.Helper()
		for d := 3000; d < 9000; d++ {
			b64 := bigB64(d)
			block := imageBlock(b64)
			body := []byte(`{"messages":[` +
				`{"role":"user","content":[` + block + `]},` +
				`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
				`]}`)
			repl := imageOmission(body, mustImageNode(t, body))
			if len(block)-len(repl) == target {
				return body
			}
		}
		t.Fatalf("no image size yields savings %d", target)
		return nil
	}
	gated := find(4095)
	opts := pressureOptions(len(gated))
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	plan, err := BuildKimiRequestPlan(gated, opts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Stats.OldImages != 0 {
		t.Fatal("edit saving 4095 bytes was not gated")
	}
	if plan.FitsKimi {
		t.Fatal("nonfitting body reported as fitting after gated edit")
	}
	accepted := find(4099)
	opts2 := pressureOptions(len(accepted))
	opts2.ProtectedMessages = 1
	opts2.ProtectedImages = 0
	plan2, err := BuildKimiRequestPlan(accepted, opts2)
	if err != nil {
		t.Fatal(err)
	}
	if plan2.Stats.OldImages != 1 {
		t.Fatal("edit saving 4099 bytes was gated")
	}
}

// TestDuplicateAnchorRemoved: when trimming removes the kept anchor, its
// duplicate markers downgrade to static text instead of pointing at nothing.
func TestDuplicateAnchorRemoved(t *testing.T) {
	for _, d := range []int{4400, 4600} {
		t.Run(fmt.Sprintf("size_%d", d), func(t *testing.T) {
			b64 := bigB64(d)
			body := []byte(`{"messages":[` +
				`{"role":"user","content":[` + imageBlock(b64) + `]},` +
				`{"role":"user","content":[` + imageBlock(b64) + `]},` +
				`{"role":"user","content":[` + imageBlock(b64) + `]},` +
				`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
				`]}`)
			opts := pressureOptions(len(body))
			opts.ProtectedMessages = 1
			opts.ProtectedImages = 0
			plan, err := BuildKimiRequestPlan(body, opts)
			if err != nil {
				t.Fatal(err)
			}
			out := string(mustApplyValid(t, body, plan))
			if plan.Stats.DuplicateImages != 2 || plan.Stats.OldImages != 1 {
				t.Fatalf("stats = %+v, want 2 dups + 1 old image", plan.Stats)
			}
			if strings.Contains(out, "[duplicate of image ") {
				t.Fatal("marker still points at removed anchor")
			}
			if strings.Count(out, "[duplicate image omitted]") != 2 {
				t.Fatalf("static markers missing: %s", out[:400])
			}
		})
	}
}

func TestNewestImagesProtectedIndependently(t *testing.T) {
	var sb strings.Builder
	sb.WriteString(`{"messages":[`)
	sb.WriteString(`{"role":"user","content":[` + imageBlock(bigB64(4400)) + `]},`)
	sb.WriteString(`{"role":"user","content":[` + imageBlock(bigB64(4500)) + `]},`)
	for i := 0; i < 26; i++ {
		fmt.Fprintf(&sb, `{"role":"user","content":[{"type":"text","text":"filler-%d"}]},`, i)
	}
	sb.WriteString(`{"role":"user","content":[{"type":"text","text":"tail"}]}`)
	sb.WriteString(`]}`)
	body := []byte(sb.String())
	opts := pressureOptions(len(body))
	opts.ProtectedImages = 2
	opts.ProtectedMessages = 24
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	if strings.Count(out, `"type":"image"`) != 2 {
		t.Fatalf("newest images not protected")
	}
}

func TestPlanOptionsBudgetValidation(t *testing.T) {
	body := toolSearchFixture()
	o := DefaultPlanOptions()
	o.LandingBytes = o.TriggerBytes // equal: invalid
	if _, err := BuildKimiRequestPlan(body, o); err == nil {
		t.Fatal("expected budget validation error")
	}
}

// TestStage1GrowthCounted: stage-1 growth is negative savings the prefix
// loop must earn back — a body that only fits after consuming every
// candidate must not be rejected early.
func TestStage1GrowthCounted(t *testing.T) {
	// Two distinct image payloads, each omitting for >= 4096 bytes.
	findSavings := func(target int) string {
		for d := 3000; d < 9000; d++ {
			b64 := bigB64(d)
			block := imageBlock(b64)
			body := imageFixtureBody(block)
			if len(block)-len(imageOmission(body, mustImageNode(t, body))) == target {
				return b64
			}
		}
		t.Fatalf("no image size yields savings %d", target)
		return ""
	}
	b64A := findSavings(4099)
	b64B := findSavings(4103)
	// 2200 compact tool_reference blocks: each replacement is 2 bytes
	// longer, so stage 1 grows the body by 4400 — more than either image
	// saves, so stopping after one candidate would be a false non-fit.
	var refs strings.Builder
	for i := 0; i < 2200; i++ {
		if i > 0 {
			refs.WriteString(",")
		}
		refs.WriteString(`{"type":"tool_reference","tool_name":"X"}`)
	}
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"g","content":[` + refs.String() + `]}]},` +
		`{"role":"user","content":[` + imageBlock(b64A) + `]},` +
		`{"role":"user","content":[` + imageBlock(b64B) + `]},` +
		`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	opts.LandingBytes = len(body) - 3000 // required savings = 3000 (quantum 1)
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustApplyValid(t, body, plan)
	if plan.Stats.OldImages != 2 {
		t.Fatalf("growth was not earned back: OldImages=%d (stats %+v)", plan.Stats.OldImages, plan.Stats)
	}
	if !plan.FitsKimi {
		t.Fatalf("fitting body rejected: estimate %d > trigger %d", plan.EstimatedBytes, opts.TriggerBytes)
	}
}

// TestStage1GrowthOnlyNonFit: a body at or below trigger whose stage-1
// normalization alone pushes the estimate over trigger is non-fitting, and
// stage 2/3 must not run for it.
func TestStage1GrowthOnlyNonFit(t *testing.T) {
	var refs strings.Builder
	for i := 0; i < 200; i++ {
		if i > 0 {
			refs.WriteString(",")
		}
		refs.WriteString(`{"type":"tool_reference","tool_name":"X"}`)
	}
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"g","content":[` + refs.String() + `]}]},` +
		`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
		`]}`)
	opts := DefaultPlanOptions()
	opts.TriggerBytes = len(body) // raw == trigger: no stage 2/3 by size
	opts.LandingBytes = 1000
	opts.QuantumBytes = 1
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	if plan.FitsKimi {
		t.Fatal("stage-1 growth over trigger should not fit")
	}
	if plan.Stats.DuplicateImages != 0 || plan.Stats.OldImages != 0 || plan.Stats.TrimmedToolResults != 0 {
		t.Fatalf("stage 2/3 ran for a raw-at-trigger body: %+v", plan.Stats)
	}
	if plan.Stats.ToolReferences != 200 {
		t.Fatalf("stage 1 refs = %d", plan.Stats.ToolReferences)
	}
}

// TestTrimSavingsGateToolResult: the 4096-byte gate applies to tool_result
// shortening at the exact boundary.
func TestTrimSavingsGateToolResult(t *testing.T) {
	find := func(target int) []byte {
		t.Helper()
		for n := 1001; n < 20000; n++ {
			body := []byte(`{"messages":[` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"g","content":"` + strings.Repeat("x", n) + `"}]},` +
				`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
				`]}`)
			root, err := parseJSON(body)
			if err != nil {
				t.Fatal(err)
			}
			tr := root.field("messages").elems[0].field("content").elems[0]
			sn := toolResultTextNodes(body, tr)[0]
			decoded, err := decodeString(body[sn.start:sn.end])
			if err != nil {
				t.Fatal(err)
			}
			head := headRunes(decoded, 50)
			tail := tailRunes(decoded, 50)
			omitted := len(decoded) - len(head) - len(tail)
			if omitted <= 0 {
				continue
			}
			marshaled, err := json.Marshal(head + "\n…[" + itoa(omitted) + " bytes omitted for length]…\n" + tail)
			if err != nil {
				t.Fatal(err)
			}
			if sn.end-sn.start-len(marshaled) == target {
				return body
			}
		}
		t.Fatalf("no result size yields savings %d", target)
		return nil
	}
	build := func(body []byte) (RequestPlan, error) {
		opts := pressureOptions(len(body))
		opts.ProtectedMessages = 1
		opts.LargeToolResultBytes = 1000
		opts.ToolResultHeadBytes = 50
		opts.ToolResultTailBytes = 50
		return BuildKimiRequestPlan(body, opts)
	}
	gated, err := build(find(4095))
	if err != nil {
		t.Fatal(err)
	}
	if gated.Stats.TrimmedToolResults != 0 {
		t.Fatal("result shortening at 4095 savings was not gated")
	}
	accepted, err := build(find(4096))
	if err != nil {
		t.Fatal(err)
	}
	if accepted.Stats.TrimmedToolResults != 1 {
		t.Fatal("result shortening at exactly 4096 savings was gated")
	}
}

// TestTrimSavingsGateDuplicateMarker: the gate applies to duplicate-image
// markers at the exact boundary, across media types to reach both residues.
func TestTrimSavingsGateDuplicateMarker(t *testing.T) {
	medias := []string{"image/png", "image/pn", "image/p", "image/pngx", "image/pngxy"}
	find := func(target int) []byte {
		t.Helper()
		for _, media := range medias {
			for d := 3000; d < 9000; d++ {
				b64 := bigB64(d)
				block := imageBlockMedia(b64, media)
				body := []byte(`{"messages":[` +
					`{"role":"user","content":[` + block + `]},` +
					`{"role":"user","content":[` + block + `]},` +
					`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
					`]}`)
				first := rootFirstImage(t, body)
				id, ok := identifyImage(body, first)
				if !ok {
					t.Fatal("identity failed")
				}
				savings := len(block) - len(duplicateImageOmission(id, true))
				if savings == target {
					return body
				}
			}
		}
		t.Fatalf("no dup marker yields savings %d", target)
		return nil
	}
	build := func(body []byte) (RequestPlan, error) {
		opts := pressureOptions(len(body))
		opts.ProtectedMessages = 1
		opts.ProtectedImages = 0
		opts.LandingBytes = len(body) - 5000 // force stage 2 to matter for fit
		opts.QuantumBytes = 1
		return BuildKimiRequestPlan(body, opts)
	}
	// Below the gate the duplicate must survive even under pressure.
	gated, err := build(find(4095))
	if err != nil {
		t.Fatal(err)
	}
	if gated.Stats.DuplicateImages != 0 {
		t.Fatal("dup marker at 4095 savings was not gated")
	}
	accepted, err := build(find(4096))
	if err != nil {
		t.Fatal(err)
	}
	if accepted.Stats.DuplicateImages != 1 {
		t.Fatal("dup marker at exactly 4096 savings was gated")
	}
	if !accepted.FitsKimi {
		t.Fatal("body should fit once the dup is replaced")
	}
}

func rootFirstImage(t *testing.T, body []byte) *jnode {
	t.Helper()
	root, err := parseJSON(body)
	if err != nil {
		t.Fatal(err)
	}
	return root.field("messages").elems[0].field("content").elems[0]
}

// TestAnchorFixupExactBoundary: anchor-fixup shortening supplies the last
// required bytes, so the next candidate in the age order is not selected.
// Runs at two image sizes.
func TestAnchorFixupExactBoundary(t *testing.T) {
	for _, d := range []int{4400, 4600} {
		t.Run(fmt.Sprintf("size_%d", d), func(t *testing.T) {
			b64 := bigB64(d)
			anchorBlock := imageBlock(b64)
			otherB64 := bigB64(d + 400)
			body := []byte(`{"messages":[` +
				`{"role":"user","content":[` + anchorBlock + `]},` + // anchor
				`{"role":"user","content":[` + anchorBlock + `]},` + // dup 1
				`{"role":"user","content":[` + anchorBlock + `]},` + // dup 2
				`{"role":"user","content":[` + imageBlock(otherB64) + `]},` + // younger unique image
				`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
				`]}`)
			// Compute expected savings with the production helpers.
			anchorNode := rootFirstImage(t, body)
			id, ok := identifyImage(body, anchorNode)
			if !ok {
				t.Fatal("identity failed")
			}
			dupSavings := len(anchorBlock) - len(duplicateImageOmission(id, true))
			anchorMarginal := len(anchorBlock) - len(imageOmission(body, anchorNode))
			fixup := len(duplicateImageOmission(id, true)) - len(duplicateImageOmission(id, false))
			total := 2*dupSavings + anchorMarginal + 2*fixup

			opts := pressureOptions(len(body))
			opts.ProtectedMessages = 1
			opts.ProtectedImages = 0
			opts.LandingBytes = len(body) - total // required == total exactly
			opts.QuantumBytes = 1
			plan, err := BuildKimiRequestPlan(body, opts)
			if err != nil {
				t.Fatal(err)
			}
			out := string(mustApplyValid(t, body, plan))
			if plan.Stats.DuplicateImages != 2 || plan.Stats.OldImages != 1 {
				t.Fatalf("stats = %+v, want 2 dups + exactly 1 old image (anchor)", plan.Stats)
			}
			if strings.Count(out, "[duplicate image omitted]") != 2 {
				t.Fatal("markers not downgraded after anchor removal")
			}
			if !strings.Contains(out, otherB64) {
				t.Fatal("next candidate was selected despite exact boundary")
			}
		})
	}
}

// TestExhaustedCandidatesStillFit: candidates running out above the landing
// target still fit when the estimate lands at or below trigger.
func TestExhaustedCandidatesStillFit(t *testing.T) {
	b64, _ := func() (string, int) {
		for d := 3000; d < 9000; d++ {
			b := bigB64(d)
			block := imageBlock(b)
			body := imageFixtureBody(block)
			if len(block)-len(imageOmission(body, mustImageNode(t, body))) == 4099 {
				return b, 0
			}
		}
		t.Fatal("no image size yields savings 4099")
		return "", 0
	}()
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[` + imageBlock(b64) + `]},` +
		`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
		`]}`)
	opts := pressureOptions(len(body)) // landing 64: requirement far beyond one edit
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustApplyValid(t, body, plan)
	if plan.Stats.OldImages != 1 {
		t.Fatalf("the one eligible candidate was not consumed: %+v", plan.Stats)
	}
	if !plan.FitsKimi {
		t.Fatalf("estimate %d <= trigger %d must fit", plan.EstimatedBytes, opts.TriggerBytes)
	}
}

// TestSameMessageAgeOrder: candidates in the same message order by source
// position (image before the later tool_result text).
func TestSameMessageAgeOrder(t *testing.T) {
	big := strings.Repeat("x", 20000)
	body := []byte(`{"messages":[` +
		`{"role":"user","content":[` + imageBlock(bigB64(4400)) + `,{"type":"tool_result","tool_use_id":"s","content":"` + big + `"}]},` +
		`{"role":"user","content":[{"type":"text","text":"tail"}]}` +
		`]}`)
	opts := pressureOptions(len(body))
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0
	opts.LandingBytes = len(body) - 4400 - 100
	opts.QuantumBytes = 65536 // force consumption of both candidates
	plan, err := BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	out := string(mustApplyValid(t, body, plan))
	imgIdx := strings.Index(out, "omitted to fit request budget")
	resIdx := strings.Index(out, "bytes omitted for length")
	if imgIdx < 0 || resIdx < 0 {
		t.Fatalf("both candidates should trim: %s", out[:300])
	}
	if imgIdx > resIdx {
		t.Fatal("same-message candidates not ordered by source position")
	}
}

// TestQuantizedSavingsDefaults: with the shipped defaults, one byte past the
// trigger demands exactly two quanta.
func TestQuantizedSavingsDefaults(t *testing.T) {
	d := DefaultPlanOptions()
	if got := quantizedSavings(d.TriggerBytes+1, d.LandingBytes, d.QuantumBytes); got != 524_288 {
		t.Fatalf("required = %d, want 524288", got)
	}
	if got := quantizedSavings(d.TriggerBytes, d.LandingBytes, d.QuantumBytes); got != 524_288 {
		t.Fatalf("required at trigger = %d, want 524288", got)
	}
}

// TestPrefixExtensionAcrossBatches: growing only the protected suffix steps
// the quantized requirement by exactly one quantum, and the next batch is a
// strict extension — batch-1 edits bit-identical, one more oldest candidate.
func TestPrefixExtensionAcrossBatches(t *testing.T) {
	find := func(target int) string {
		for d := 3000; d < 9000; d++ {
			b64 := bigB64(d)
			block := imageBlock(b64)
			body := imageFixtureBody(block)
			if len(block)-len(imageOmission(body, mustImageNode(t, body))) == target {
				return b64
			}
		}
		t.Fatalf("no image size yields savings %d", target)
		return ""
	}
	const s = 4099 // per-candidate savings; doubles as the quantum
	pads := []string{find(s), find(s + 4), find(s + 8), find(s + 12)}
	media := func() string {
		var sb strings.Builder
		for _, p := range pads {
			sb.WriteString(`{"role":"user","content":[` + imageBlock(p) + `]},`)
		}
		return sb.String()
	}
	base := []byte(`{"messages":[` + media() + `{"role":"user","content":[{"type":"text","text":"tail"}]}]}`)

	opts := DefaultPlanOptions()
	opts.TriggerBytes = len(base) - 1
	opts.LandingBytes = len(base) - s // required = s -> one candidate
	opts.QuantumBytes = s
	opts.ProtectedMessages = 1
	opts.ProtectedImages = 0

	plan1, err := BuildKimiRequestPlan(base, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustApplyValid(t, base, plan1)
	if plan1.Stats.OldImages != 1 {
		t.Fatalf("batch 1 = %d images, want 1", plan1.Stats.OldImages)
	}

	// Grow only the protected suffix by less than one quantum's shortfall
	// step: required goes from s to 2s.
	grown := []byte(`{"messages":[` + media() +
		`{"role":"user","content":[{"type":"text","text":"tail"}]},` +
		`{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("g", s/2) + `"}]}` +
		`]}`)
	plan2, err := BuildKimiRequestPlan(grown, opts)
	if err != nil {
		t.Fatal(err)
	}
	mustApplyValid(t, grown, plan2)
	if plan2.Stats.OldImages != 2 {
		t.Fatalf("batch 2 = %d images, want 2 (stats %+v)", plan2.Stats.OldImages, plan2.Stats)
	}
	// Strict extension: batch-1 edits bit-identical, one more oldest candidate.
	if len(plan2.Edits) != len(plan1.Edits)+1 {
		t.Fatalf("edits %d != %d+1", len(plan2.Edits), len(plan1.Edits))
	}
	for i := range plan1.Edits {
		if !reflect.DeepEqual(plan1.Edits[i], plan2.Edits[i]) {
			t.Fatalf("batch-1 edit %d changed across batches: %+v vs %+v", i, plan1.Edits[i], plan2.Edits[i])
		}
	}
}
