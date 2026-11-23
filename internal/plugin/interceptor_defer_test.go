package plugin

// Tests for deferring request-body decoding: the body of a syntactically valid
// interceptor envelope is decoded only once an eligibility check has passed.
// Skipped callbacks ignore a malformed body; eligible callbacks still fail.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

// interceptWire builds an interceptor envelope with a verbatim Body JSON
// literal, so tests can send bodies that encoding/json would refuse to decode
// into []byte.
func interceptWire(t *testing.T, requestID, sourceFormat string, headers map[string][]string, metadata map[string]any, bodyJSON string) []byte {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"RequestID":    requestID,
		"SourceFormat": sourceFormat,
		"Headers":      headers,
		"Metadata":     metadata,
		"Body":         json.RawMessage(bodyJSON),
	})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestBeforeAuthIgnoresMalformedBodyWhenNestedOnly(t *testing.T) {
	configureTest(t, "nested_only: true\n")
	wire := interceptWire(t, "req-bad", "claude", nil, nil, `"!!!"`)
	raw, err := interceptBeforeAuth(wire)
	if err != nil {
		t.Fatalf("ineligible callback must not decode the body: %v", err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.Headers) != 0 || len(resp.BodyEdits) != 0 || resp.Terminate {
		t.Fatalf("outer pass must be a no-op: %+v", resp)
	}
}

func TestBeforeAuthIgnoresNonStringBodyWhenNonClaude(t *testing.T) {
	configureTest(t, "")
	wire := interceptWire(t, "req-bad", "openai", map[string][]string{reservationHeader: {"res-x"}}, nil, `12345`)
	raw, err := interceptBeforeAuth(wire)
	if err != nil {
		t.Fatalf("ineligible callback must not decode the body: %v", err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.Headers) != 0 || len(resp.BodyEdits) != 0 {
		t.Fatalf("non-claude request not skipped: %+v", resp)
	}
}

func TestAfterAuthIgnoresMalformedBodyWhenNonKimi(t *testing.T) {
	configureTest(t, "")
	stubAuthProvider("claude")
	defer func() { HostCall = nil }()
	wire := interceptWire(t, "req-bad", "claude", map[string][]string{reservationHeader: {"res-x"}}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-nk",
	}, `"!!!"`)
	raw, err := interceptAfterAuth(wire)
	if err != nil {
		t.Fatalf("non-kimi callback must not decode the body: %v", err)
	}
	resp := decodeInterceptResponse(t, raw)
	if len(resp.BodyEdits) != 0 || resp.Body != nil || resp.Terminate {
		t.Fatalf("non-kimi body touched: %+v", resp)
	}
}

// TestEligibleMalformedBase64StillErrors: once format, nested-only, and
// provider eligibility are satisfied, a body that cannot be decoded is an
// error, as it was when the whole envelope was decoded up front.
func TestEligibleMalformedBase64StillErrors(t *testing.T) {
	configureTest(t, "")
	stubAuthProvider("kimi")
	defer func() { HostCall = nil }()

	before := interceptWire(t, "req-bad", "claude", map[string][]string{reservationHeader: {"res-eligible"}}, map[string]any{
		"source": "plugin_host_model_callback",
	}, `"!!!"`)
	if _, err := interceptBeforeAuth(before); err == nil {
		t.Fatal("before-auth must reject an eligible undecodable body")
	}

	after := interceptWire(t, "req-bad", "claude", map[string][]string{reservationHeader: {"res-eligible"}}, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-k",
	}, `"!!!"`)
	if _, err := interceptAfterAuth(after); err == nil {
		t.Fatal("after-auth must reject an eligible undecodable body")
	}
}

// TestInterceptIgnoresNonStringBodyWhenEligible: a body that is valid JSON but
// not a string fails decoding at the eligible site, matching the old
// envelope-level failure.
func TestInterceptIgnoresNonStringBodyWhenEligible(t *testing.T) {
	configureTest(t, "")
	wire := interceptWire(t, "req-bad", "claude", nil, nil, `{"not":"a string"}`)
	if _, err := interceptBeforeAuth(wire); err == nil {
		t.Fatal("eligible non-string body must fail decoding")
	}
}

// TestTraceBodyBytesAccurateForSkippedCallback: trace output must report the
// decoded body size even when the callback skips before decoding, so traces
// stay comparable with the eager-decode behavior.
func TestTraceBodyBytesAccurateForSkippedCallback(t *testing.T) {
	traceFile := filepath.Join(t.TempDir(), "trace.jsonl")
	configureTest(t, "trace_file: "+traceFile+"\n")
	body := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	wire, err := json.Marshal(pluginapi.RequestInterceptRequest{
		RequestID:    "req-trace",
		SourceFormat: "openai",
		Body:         body,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := interceptBeforeAuth(wire); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(traceFile)
	if err != nil {
		t.Fatal(err)
	}
	var line map[string]string
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &line); err != nil {
		t.Fatalf("decode trace line %q: %v", raw, err)
	}
	want := jsonNumber(len(body))
	if line["body_bytes"] != want {
		t.Fatalf("body_bytes = %q, want %q", line["body_bytes"], want)
	}
}

func jsonNumber(n int) string {
	raw, _ := json.Marshal(n)
	return string(raw)
}

// TestTraceEnabledMalformedBodyStillErrors: tracing needs the decoded size, so
// a malformed body fails while tracing is on, as before.
func TestTraceEnabledMalformedBodyStillErrors(t *testing.T) {
	traceFile := filepath.Join(t.TempDir(), "trace.jsonl")
	configureTest(t, "trace_file: "+traceFile+"\n")
	wire := interceptWire(t, "req-bad", "openai", nil, nil, `"!!!"`)
	if _, err := interceptBeforeAuth(wire); err == nil {
		t.Fatal("trace-enabled callback must fail on an undecodable body")
	}
}

// TestBodyDecodeMatchesStdlib pins the deferred decode to encoding/json's
// semantics for every body shape that used to be decoded by envelope parsing.
func TestBodyDecodeMatchesStdlib(t *testing.T) {
	cases := []string{
		`""`,
		`"aGVsbG8="`,
		`"aGVsbG8"`,   // missing padding
		`"aGVs bG8="`, // embedded space
		`"aGVsbG8=="`, // excess padding
		`"!!!!"`,
		`null`,
		`123`,
		`true`,
		`{}`,
		`[]`,
	}
	for _, bodyJSON := range cases {
		t.Run(bodyJSON, func(t *testing.T) {
			wire := interceptWire(t, "req", "claude", nil, nil, bodyJSON)
			req, err := decodeInterceptRequest(wire, false)
			if err != nil {
				t.Fatalf("envelope decode: %v", err)
			}
			got, gotErr := req.decodeBody()

			var legacy pluginapi.RequestInterceptRequest
			legacyErr := json.Unmarshal(wire, &legacy)

			if (gotErr == nil) != (legacyErr == nil) {
				t.Fatalf("decode error mismatch: deferred=%v legacy=%v", gotErr, legacyErr)
			}
			if gotErr == nil && string(got) != string(legacy.Body) {
				t.Fatalf("body mismatch: deferred=%q legacy=%q", got, legacy.Body)
			}
			if gotErr == nil && legacy.Body == nil && got != nil && len(got) != 0 {
				t.Fatalf("empty body became %q", got)
			}
		})
	}
}
