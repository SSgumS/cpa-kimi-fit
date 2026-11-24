package plugin

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
	"github.com/router-for-me/kimi-fit-plugin/internal/planner"
)

func configureTest(t *testing.T, yaml string) {
	t.Helper()
	Shutdown()
	if err := configure(lifecycleRequest{ConfigYAML: []byte(yaml)}); err != nil {
		t.Fatalf("configure: %v", err)
	}
	t.Cleanup(Shutdown)
}

func stubAuthProvider(provider string) {
	HostCall = func(method string, payload []byte) ([]byte, int, error) {
		resp := pluginapi.HostAuthGetRuntimeResponse{
			Auth: pluginapi.HostAuthFileEntry{Provider: provider},
		}
		result, _ := json.Marshal(resp)
		env, _ := json.Marshal(envelope{OK: true, Result: result})
		return env, 0, nil
	}
}

func interceptReq(t *testing.T, sourceFormat string, headers map[string][]string, metadata map[string]any, body []byte) []byte {
	t.Helper()
	raw, err := json.Marshal(pluginapi.RequestInterceptRequest{
		RequestID:    "req-test",
		SourceFormat: sourceFormat,
		Headers:      headers,
		Metadata:     metadata,
		Body:         body,
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func decodeInterceptResponse(t *testing.T, raw []byte) pluginapi.RequestInterceptResponse {
	t.Helper()
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("decode envelope: %v", err)
	}
	if !env.OK {
		t.Fatalf("error envelope: %+v", env.Error)
	}
	var resp pluginapi.RequestInterceptResponse
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return resp
}

var toolRefBody = []byte(`{"model":"k3.1-pro","messages":[` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` +
	`{"type":"tool_reference","tool_name":"WebSearch"}]}]},` +
	`{"role":"user","content":[{"type":"text","text":"latest"}]}` +
	`]}`)

func TestBeforeAuthSkipsNonClaude(t *testing.T) {
	configureTest(t, "")
	raw, err := interceptBeforeAuth(interceptReq(t, "openai", map[string][]string{
		reservationHeader: {"res-1"},
	}, nil, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.Headers) != 0 || len(resp.BodyEdits) != 0 {
		t.Fatalf("non-claude request not skipped: %+v", resp)
	}
}

func TestBeforeAuthPlansWithoutBodyMutation(t *testing.T) {
	configureTest(t, "")
	raw, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-plan"},
	}, map[string]any{"source": "plugin_host_model_callback"}, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.BodyEdits) != 0 || resp.Body != nil {
		t.Fatalf("before-auth must not edit body: %+v", resp)
	}
	if len(resp.Headers) != 0 {
		t.Fatalf("fitting plan emitted exclusion: %+v", resp.Headers)
	}
	if s := planStoreInstance(); s == nil || s.len() != 1 {
		t.Fatalf("plan not cached")
	}
}

func oversizedBody(t *testing.T, target int) []byte {
	t.Helper()
	big := strings.Repeat("x", target*2)
	return []byte(`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"` + big + `"}]}]}`)
}

func TestBeforeAuthNonFittingEmitsExclusionWithReservation(t *testing.T) {
	configureTest(t, "length_optimization_enabled: true\ntrigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\n")
	body := oversizedBody(t, 1000)
	raw, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-big"},
	}, map[string]any{"source": "plugin_host_model_callback"}, body))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	got := resp.Headers.Get(excludeProvidersHeader)
	if got != kimiExclusionValue {
		t.Fatalf("exclusion header = %q, want %q", got, kimiExclusionValue)
	}
}

func TestBeforeAuthNonFittingSilentWithoutReservation(t *testing.T) {
	configureTest(t, "length_optimization_enabled: true\ntrigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\n")
	body := oversizedBody(t, 1000)
	raw, err := interceptBeforeAuth(interceptReq(t, "claude", nil, nil, body))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if got := resp.Headers.Get(excludeProvidersHeader); got != "" {
		t.Fatalf("untrusted exclusion emitted: %q", got)
	}
	// Plan still cached under request ID for a direct (non-nested) route.
	if s := planStoreInstance(); s == nil || s.len() != 1 {
		t.Fatal("plan not cached under request key")
	}
}

func TestBeforeAuthNestedOnlySkipsOuterPass(t *testing.T) {
	configureTest(t, "nested_only: true\n")
	raw, err := interceptBeforeAuth(interceptReq(t, "claude", nil, nil, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.Headers) != 0 {
		t.Fatalf("outer pass should be a no-op: %+v", resp)
	}
	if s := planStoreInstance(); s != nil && s.len() != 0 {
		t.Fatal("outer pass created a plan despite nested_only")
	}
}

func TestAfterAuthNonKimiPreservesBody(t *testing.T) {
	configureTest(t, "")
	stubAuthProvider("claude")
	defer func() { HostCall = nil }()
	// Seed a plan; non-Kimi must ignore it entirely.
	rawBefore, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-nk"},
	}, map[string]any{"source": "plugin_host_model_callback"}, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	_ = rawBefore
	raw, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-nk"},
	}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-1",
	}, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.BodyEdits) != 0 || resp.Body != nil || resp.Terminate {
		t.Fatalf("non-kimi body touched: %+v", resp)
	}
	if len(resp.ClearHeaders) == 0 || resp.ClearHeaders[0] != excludeProvidersHeader {
		t.Fatalf("exclusion header not cleared: %v", resp.ClearHeaders)
	}
}

func TestAfterAuthKimiAppliesPlanEdits(t *testing.T) {
	configureTest(t, "")
	stubAuthProvider("kimi")
	defer func() { HostCall = nil }()
	if _, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-k"},
	}, map[string]any{"source": "plugin_host_model_callback"}, toolRefBody)); err != nil {
		t.Fatal(err)
	}
	raw, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-k"},
	}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-k",
	}, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.BodyEdits) != 1 {
		t.Fatalf("edits = %d, want 1 tool_reference replacement", len(resp.BodyEdits))
	}
	e := resp.BodyEdits[0]
	if !strings.Contains(e.Replacement, "WebSearch") {
		t.Fatalf("replacement lost tool name: %s", e.Replacement)
	}
	if e.Start >= e.End || toolRefBody[e.Start] != '{' {
		t.Fatalf("bad edit range %d..%d", e.Start, e.End)
	}
}

func TestAfterAuthKimiNonFittingReturns413(t *testing.T) {
	configureTest(t, "length_optimization_enabled: true\ntrigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\n")
	stubAuthProvider("kimi")
	defer func() { HostCall = nil }()
	body := oversizedBody(t, 1000)
	if _, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-413"},
	}, map[string]any{"source": "plugin_host_model_callback"}, body)); err != nil {
		t.Fatal(err)
	}
	raw, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-413"},
	}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-k",
	}, body))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if !resp.Terminate || resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected terminate 413: %+v", resp)
	}
	if !strings.Contains(string(resp.ResponseBody), `"request_too_large"`) {
		t.Fatalf("bad 413 body: %s", resp.ResponseBody)
	}
}

func TestAfterAuthSkipsWithoutSelectedAuth(t *testing.T) {
	configureTest(t, "")
	raw, err := interceptAfterAuth(interceptReq(t, "claude", nil, map[string]any{
		"source": "plugin_host_model_callback",
	}, toolRefBody))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.BodyEdits) != 0 || resp.Terminate {
		t.Fatalf("outer pass must be a no-op: %+v", resp)
	}
}

func TestCompleteRequestDeletesPlan(t *testing.T) {
	configureTest(t, "")
	if _, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-done"},
	}, map[string]any{"source": "plugin_host_model_callback"}, toolRefBody)); err != nil {
		t.Fatal(err)
	}
	if s := planStoreInstance(); s == nil || s.len() != 1 {
		t.Fatal("plan missing")
	}
	completion, _ := json.Marshal(pluginapi.RequestCompletion{
		RequestID: "req-test",
		Outcome:   pluginapi.RequestCompletionSucceeded,
	})
	if _, err := completeRequest(completion); err != nil {
		t.Fatal(err)
	}
	if s := planStoreInstance(); s == nil || s.len() != 0 {
		t.Fatal("plan not deleted on completion")
	}
}

// oversizedResultsBody builds a body with a protected last message of
// protectedBytes and older large tool_results totaling unprotectedBytes.
func oversizedResultsBody(protectedBytes, unprotectedBytes int) []byte {
	var sb strings.Builder
	sb.WriteString(`{"messages":[`)
	first := true
	for unprotectedBytes > 0 {
		n := unprotectedBytes
		if n > 40000 {
			n = 40000
		}
		if !first {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"u","content":"` + strings.Repeat("x", n) + `"}]}`)
		unprotectedBytes -= n
	}
	// Small fillers push the unprotected results out of the trailing
	// 24-message window so they are actually eligible.
	for i := 0; i < 30; i++ {
		if !first || i > 0 {
			sb.WriteString(",")
		}
		first = false
		sb.WriteString(`{"role":"user","content":[{"type":"text","text":"filler"}]}`)
	}
	sb.WriteString(`,{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("p", protectedBytes) + `"}]}`)
	sb.WriteString(`]}`)
	return []byte(sb.String())
}

// TestAfterAuthKimiNonFitting413Wording: between trigger and hard limit the
// message names the budget; above the hard limit it names the hard limit —
// decided by the FINAL estimate, on both the cached-plan and inline paths.
func TestAfterAuthKimiNonFitting413Wording(t *testing.T) {
	// Defaults: trigger 1,966,080; hard limit 2,097,152.
	cases := []struct {
		name       string
		protected  int
		unprotsize int
		wantText   string
	}{
		{"budget wording above trigger", 2_000_000, 0, "exceeds the request budget"},
		{"budget wording despite raw over hard limit", 2_000_000, 200_000, "exceeds the request budget"},
		{"hard limit wording", 2_200_000, 0, "exceeds Kimi's hard limit"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, cached := range []bool{true, false} {
				configureTest(t, "length_optimization_enabled: true\n")
				stubAuthProvider("kimi")
				body := oversizedResultsBody(tc.protected, tc.unprotsize)
				if cached {
					if _, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
						reservationHeader: {"res-413w"},
					}, map[string]any{"source": "plugin_host_model_callback"}, body)); err != nil {
						t.Fatal(err)
					}
				}
				raw, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
					reservationHeader: {"res-413w"},
				}, map[string]any{
					"source":              "plugin_host_model_callback",
					"selected_auth_index": "idx-k",
				}, body))
				if err != nil {
					t.Fatal(err)
				}
				resp := decodeInterceptResponse(t, raw)
				if !resp.Terminate || resp.StatusCode != http.StatusRequestEntityTooLarge {
					t.Fatalf("cached=%v: expected terminate 413: %+v", cached, resp)
				}
				if !strings.Contains(string(resp.ResponseBody), tc.wantText) {
					t.Fatalf("cached=%v: body missing %q: %s", cached, tc.wantText, resp.ResponseBody)
				}
				if !strings.Contains(string(resp.ResponseBody), `"request_too_large"`) {
					t.Fatalf("cached=%v: bad 413 body: %s", cached, resp.ResponseBody)
				}
			}
		})
	}
}

// TestInterceptorsFailOpenOnMalformedBody: a body that fails planning must
// pass through untouched — no cached plan, no exclusion, no edits, no 413.
func TestInterceptorsFailOpenOnMalformedBody(t *testing.T) {
	configureTest(t, "length_optimization_enabled: true\ntrigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\n")
	stubAuthProvider("kimi")
	defer func() { HostCall = nil }()
	malformed := append([]byte(`{"messages":[{broken`), []byte(strings.Repeat("x", 2000))...)

	before, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-bad"},
	}, map[string]any{"source": "plugin_host_model_callback"}, malformed))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, before)
	if len(resp.Headers) != 0 || len(resp.BodyEdits) != 0 || resp.Terminate {
		t.Fatalf("before-auth mutated on malformed body: %+v", resp)
	}
	if s := planStoreInstance(); s != nil && s.len() != 0 {
		t.Fatal("malformed body cached a plan")
	}

	after, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-bad"},
	}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-k",
	}, malformed))
	if err != nil {
		t.Fatal(err)
	}
	resp = decodeInterceptResponse(t, after)
	if len(resp.BodyEdits) != 0 || resp.Terminate || resp.Body != nil {
		t.Fatalf("after-auth mutated on malformed body: %+v", resp)
	}
	if len(resp.ClearHeaders) == 0 || resp.ClearHeaders[0] != excludeProvidersHeader {
		t.Fatalf("exclusion header not cleared: %v", resp.ClearHeaders)
	}
}

// trimPressureFixture builds a body above the default trigger that carries
// every trim candidate: an eligible duplicate image pair, a large old
// tool_result, and a tool_reference block. Two newer distinct images in the
// protected tail consume ProtectedImages so the old pair stays eligible. Any
// size edit is therefore detectable, while the tool_reference rewrite is the
// only edit that must survive with trimming off.
func trimPressureFixture(t *testing.T) []byte {
	t.Helper()
	oldImg := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` +
		strings.Repeat("QUJD", 1100) + `"}}` // ~4.4 KiB decoded
	tailImgA := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` +
		strings.Repeat("REVG", 1100) + `"}}`
	tailImgB := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` +
		strings.Repeat("R0hJ", 1100) + `"}}`
	var sb strings.Builder
	sb.WriteString(`{"model":"k3.1-pro","messages":[`)
	sb.WriteString(`{"role":"user","content":[` + oldImg + `]},`)
	sb.WriteString(`{"role":"user","content":[` + oldImg + `]},`)
	sb.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"` +
		strings.Repeat("y", 100_000) + `"}]},`)
	sb.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":[` +
		`{"type":"tool_reference","tool_name":"WebSearch"}]}]},`)
	// Fillers push the candidates out of the 24-message protection window.
	for i := 0; i < 30; i++ {
		sb.WriteString(`{"role":"user","content":[{"type":"text","text":"filler"}]},`)
	}
	// Padding lands in the protected tail so trimming alone cannot fix the size.
	sb.WriteString(`{"role":"user","content":[{"type":"text","text":"` +
		strings.Repeat("p", 2_000_000) + `"}]},`)
	sb.WriteString(`{"role":"user","content":[` + tailImgA + `]},`)
	sb.WriteString(`{"role":"user","content":[` + tailImgB + `]}`)
	sb.WriteString(`]}`)
	return []byte(sb.String())
}

func authMeta() map[string]any {
	return map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-k",
	}
}

// TestLengthOptimizationDisabledByDefault: with the default config the same
// oversized body yields no exclusion and no 413; its only edit is the
// tool_reference text replacement. The trimming-enabled control on the same
// body terminates 413, proving the fixture would exercise the trim path.
func TestLengthOptimizationDisabledByDefault(t *testing.T) {
	body := trimPressureFixture(t)

	configureTest(t, "")
	stubAuthProvider("kimi")
	defer func() { HostCall = nil }()

	before, err := interceptBeforeAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-off"},
	}, map[string]any{"source": "plugin_host_model_callback"}, body))
	if err != nil {
		t.Fatal(err)
	}
	if got := decodeInterceptResponse(t, before).Headers.Get(excludeProvidersHeader); got != "" {
		t.Fatalf("exclusion emitted while length optimization disabled: %q", got)
	}

	after, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-off"},
	}, authMeta(), body))
	if err != nil {
		t.Fatal(err)
	}
	resp := decodeInterceptResponse(t, after)
	if resp.Terminate || resp.Body != nil {
		t.Fatalf("oversized body terminated/rewritten while optimization disabled: %+v", resp)
	}
	if len(resp.BodyEdits) != 1 {
		t.Fatalf("edits = %d, want only the tool_reference rewrite", len(resp.BodyEdits))
	}
	if e := resp.BodyEdits[0]; !strings.Contains(e.Replacement, "tool_reference: WebSearch") {
		t.Fatalf("unexpected edit replacement: %.80s", e.Replacement)
	}

	// Enabled control: the same fixture must actually exercise both trim
	// stages (a 413 alone would not prove either ran), then terminate 413
	// because the protected tail cannot be trimmed below the trigger.
	enabledOpts := planner.DefaultPlanOptions()
	enabledPlan, err := planner.BuildKimiRequestPlan(body, enabledOpts)
	if err != nil {
		t.Fatal(err)
	}
	if enabledPlan.Stats.DuplicateImages < 1 || enabledPlan.Stats.TrimmedToolResults < 1 {
		t.Fatalf("enabled control did not exercise trim stages: %+v", enabledPlan.Stats)
	}

	configureTest(t, "length_optimization_enabled: true\n")
	control, err := interceptAfterAuth(interceptReq(t, "claude", map[string][]string{
		reservationHeader: {"res-ctl"},
	}, authMeta(), body))
	if err != nil {
		t.Fatal(err)
	}
	if cresp := decodeInterceptResponse(t, control); !cresp.Terminate || cresp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("enabled control should hit the trim path and 413: %+v", cresp)
	}
}

// TestLengthOptimizationDisabledKeepsFitVerdict: disabled trimming must never
// produce a non-fitting verdict, or the exclusion/413 paths would fire on
// size alone.
func TestLengthOptimizationDisabledKeepsFitVerdict(t *testing.T) {
	body := trimPressureFixture(t)
	opts := planner.DefaultPlanOptions()
	opts.DisableLengthTrim = true
	plan, err := planner.BuildKimiRequestPlan(body, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.FitsKimi {
		t.Fatal("FitsKimi must stay true when length trimming is disabled")
	}
	if plan.Stats.DuplicateImages != 0 || plan.Stats.TrimmedToolResults != 0 || plan.Stats.OldImages != 0 {
		t.Fatalf("trim stages ran while disabled: %+v", plan.Stats)
	}
	if plan.Stats.ToolReferences != 1 {
		t.Fatalf("tool_reference normalization must still run, got %d", plan.Stats.ToolReferences)
	}
}
