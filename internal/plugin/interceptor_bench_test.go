package plugin

// Benchmarks for the request interceptor callbacks. They pin the cost of
// skipped callbacks (no decoding of multi-MB bodies) against callbacks that
// must decode and plan, so a body-decode deferral cannot silently regress the
// active paths. Run: go test -run '^$' -bench 'Intercept' -benchmem ./internal/plugin

import (
	"crypto/sha256"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

// benchRawBytes fills a deterministic pseudo-random buffer so every run builds
// the same synthetic multi-MB body without storing fixtures.
func benchRawBytes(seed string, n int) []byte {
	out := make([]byte, 0, n)
	for i := 0; len(out) < n; i++ {
		sum := sha256.Sum256([]byte(seed + strconv.Itoa(i)))
		out = append(out, sum[:]...)
	}
	return out[:n]
}

// benchMessagesBody builds a Claude Messages body of roughly targetDecoded
// bytes. withToolRef adds a nested tool_reference block, which makes the body
// eligible for planning even below the byte trigger.
func benchMessagesBody(targetDecoded int, withToolRef bool) []byte {
	const textChunk = 120 << 10
	var sb strings.Builder
	sb.Grow(targetDecoded + 4096)
	sb.WriteString(`{"model":"k3.1-pro","messages":[`)
	if withToolRef {
		sb.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` +
			`{"type":"tool_reference","tool_name":"WebSearch"}]}]},`)
	}
	filler := benchRawBytes("filler", textChunk)
	for sb.Len() < targetDecoded {
		sb.WriteString(`{"role":"user","content":[{"type":"text","text":"`)
		for _, b := range filler {
			sb.WriteByte('a' + b%26)
		}
		sb.WriteString(`"}]},`)
	}
	sb.WriteString(`{"role":"user","content":[{"type":"text","text":"latest"}]}]}`)
	return []byte(sb.String())
}

func benchWireRequest(b *testing.B, sourceFormat string, headers map[string][]string, metadata map[string]any, body []byte) []byte {
	b.Helper()
	raw, err := json.Marshal(pluginapi.RequestInterceptRequest{
		RequestID:    "bench-req",
		SourceFormat: sourceFormat,
		Headers:      headers,
		Metadata:     metadata,
		Body:         body,
	})
	if err != nil {
		b.Fatal(err)
	}
	return raw
}

func benchConfigure(b *testing.B, yaml string) {
	b.Helper()
	Shutdown()
	if err := configure(lifecycleRequest{ConfigYAML: []byte(yaml)}); err != nil {
		b.Fatalf("configure: %v", err)
	}
	b.Cleanup(Shutdown)
}

func benchStubAuthProvider(b *testing.B, provider string) {
	b.Helper()
	HostCall = func(method string, payload []byte) ([]byte, int, error) {
		resp, _ := json.Marshal(pluginapi.HostAuthGetRuntimeResponse{
			Auth: pluginapi.HostAuthFileEntry{Provider: provider},
		})
		env, _ := json.Marshal(envelope{OK: true, Result: resp})
		return env, 0, nil
	}
	b.Cleanup(func() { HostCall = nil })
}

func benchDecodeEnvelope(b *testing.B, raw []byte) {
	b.Helper()
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		b.Fatalf("decode envelope: %v", err)
	}
	if !env.OK {
		b.Fatalf("error envelope: %+v", env.Error)
	}
}

// BenchmarkInterceptBeforeAuthNestedOnlySkip is the nested-executor outer pass:
// the callback must reject the request without decoding the multi-MB body.
func BenchmarkInterceptBeforeAuthNestedOnlySkip(b *testing.B) {
	benchConfigure(b, "nested_only: true\n")
	body := benchMessagesBody(3_200_000, true)
	wire := benchWireRequest(b, "claude", nil, nil, body)
	b.ReportAllocs()
	b.SetBytes(int64(len(wire)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := interceptBeforeAuth(wire)
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			benchDecodeEnvelope(b, raw)
		}
	}
}

// BenchmarkInterceptBeforeAuthNonClaudeSkip covers the other pre-decode
// exclusion: a foreign source format.
func BenchmarkInterceptBeforeAuthNonClaudeSkip(b *testing.B) {
	benchConfigure(b, "")
	body := benchMessagesBody(3_200_000, true)
	wire := benchWireRequest(b, "openai", map[string][]string{
		reservationHeader: {"res-bench"},
	}, map[string]any{"source": "plugin_host_model_callback"}, body)
	b.ReportAllocs()
	b.SetBytes(int64(len(wire)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := interceptBeforeAuth(wire)
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			benchDecodeEnvelope(b, raw)
		}
	}
}

// BenchmarkInterceptAfterAuthNonKimiSkip: auth resolution runs before the body
// is needed, so a non-Kimi credential must not pay for decoding.
func BenchmarkInterceptAfterAuthNonKimiSkip(b *testing.B) {
	benchConfigure(b, "")
	benchStubAuthProvider(b, "claude")
	body := benchMessagesBody(3_200_000, true)
	metadata := map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-bench",
	}
	headers := map[string][]string{reservationHeader: {"res-bench"}}
	wire := benchWireRequest(b, "claude", headers, metadata, body)
	// Warm the provider cache so the benchmark measures the skip, not the host
	// callback.
	if _, err := interceptAfterAuth(wire); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(wire)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := interceptAfterAuth(wire)
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			benchDecodeEnvelope(b, raw)
		}
	}
}

// BenchmarkInterceptAfterAuthKimiActive is the eligible after-auth path: decode,
// verify the cached plan fingerprint, and return its edits.
func BenchmarkInterceptAfterAuthKimiActive(b *testing.B) {
	benchConfigure(b, "")
	benchStubAuthProvider(b, "kimi")
	body := benchMessagesBody(3_200_000, true)
	headers := map[string][]string{reservationHeader: {"res-bench"}}
	beforeWire := benchWireRequest(b, "claude", headers, map[string]any{"source": "plugin_host_model_callback"}, body)
	if _, err := interceptBeforeAuth(beforeWire); err != nil {
		b.Fatal(err)
	}
	wire := benchWireRequest(b, "claude", headers, map[string]any{
		"source":              "plugin_host_model_callback",
		"selected_auth_index": "idx-bench",
	}, body)
	b.ReportAllocs()
	b.SetBytes(int64(len(wire)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := interceptAfterAuth(wire)
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			benchDecodeEnvelope(b, raw)
		}
	}
}

// BenchmarkInterceptBeforeAuthSmall is the small-body path: decode, scan, skip.
func BenchmarkInterceptBeforeAuthSmall(b *testing.B) {
	benchConfigure(b, "")
	body := []byte(`{"model":"k3.1-pro","messages":[{"role":"user","content":[{"type":"text","text":"` +
		strings.Repeat("s", 1200) + `"}]}]}`)
	wire := benchWireRequest(b, "claude", nil, map[string]any{"source": "plugin_host_model_callback"}, body)
	b.ReportAllocs()
	b.SetBytes(int64(len(wire)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		raw, err := interceptBeforeAuth(wire)
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			benchDecodeEnvelope(b, raw)
		}
	}
}
