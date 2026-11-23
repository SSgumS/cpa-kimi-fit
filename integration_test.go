package main_test

// End-to-end harness: builds the real CPA binary plus both native plugins,
// then drives /v1/messages requests through credit-manager's nested execution
// against loopback fake upstreams. No real credentials or external calls.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	pluginExt  = ".so"
	clientKey  = "tk-abcdefg-abcdefg"
	mgmtSecret = "e2e-mgmt-secret"
	modelAlias = "high"
)

func envOrDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func resolveE2EPaths() (cpaRepo, creditRepo, tmpRoot string) {
	cpaRepo = envOrDefault("CPA_E2E_CPA_REPO", filepath.Join("..", "CLIProxyAPI"))
	creditRepo = envOrDefault("CPA_E2E_CREDIT_MANAGER_REPO", filepath.Join("..", "credit-manager"))
	tmpRoot = envOrDefault("CPA_E2E_TMPDIR", envOrDefault("GOTMPDIR", envOrDefault("TMPDIR", os.TempDir())))
	return cpaRepo, creditRepo, tmpRoot
}

var (
	buildOnce sync.Once
	buildErr  error
	binDir    string
)

type fakeUpstream struct {
	srv *httptest.Server

	mu       sync.Mutex
	bodies   [][]byte
	failNext int // number of upcoming requests to answer with failStatus
	status   int
	stream   bool
	failBody string
}

func newFakeUpstream(t *testing.T) *fakeUpstream {
	t.Helper()
	f := &fakeUpstream{status: http.StatusOK}
	f.srv = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeUpstream) url() string { return f.srv.URL }

func (f *fakeUpstream) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	failed := f.failNext > 0
	if failed {
		f.failNext--
	}
	status, failBody := f.status, f.failBody
	f.mu.Unlock()
	if failed {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(failBody))
		return
	}
	var probe struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(body, &probe)
	if probe.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_fake\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"fake\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n")
		fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n")
		fmt.Fprint(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n")
		fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"id":"msg_fake","type":"message","role":"assistant","model":"fake","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
}

func (f *fakeUpstream) requestCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.bodies)
}

func (f *fakeUpstream) lastBody() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.bodies) == 0 {
		return nil
	}
	return f.bodies[len(f.bodies)-1]
}

func (f *fakeUpstream) failOnce(status int, body string) {
	f.mu.Lock()
	f.failNext = 1
	f.status = status
	f.failBody = body
	f.mu.Unlock()
}

func buildArtifacts(t *testing.T) {
	buildOnce.Do(func() {
		cpaRepo, creditRepo, tmpRoot := resolveE2EPaths()
		if err := os.MkdirAll(tmpRoot, 0o755); err != nil {
			buildErr = err
			return
		}
		binDir, buildErr = os.MkdirTemp(tmpRoot, "kimi-fit-e2e-*")
		if buildErr != nil {
			return
		}
		pluginDir := filepath.Join(binDir, "plugins", runtime.GOOS, runtime.GOARCH)
		if buildErr = os.MkdirAll(pluginDir, 0o755); buildErr != nil {
			return
		}
		steps := [][]string{
			{"go", "build", "-o", filepath.Join(binDir, "cli-proxy-api"), "./cmd/server"},
			{"go", "build", "-buildmode=c-shared", "-o", filepath.Join(pluginDir, "credit-manager"+pluginExt), "."},
			{"go", "build", "-buildmode=c-shared", "-o", filepath.Join(pluginDir, "kimi-fit"+pluginExt), "."},
		}
		dirs := []string{cpaRepo, creditRepo, pluginRootDir(t)}
		env := append(os.Environ(),
			"NO_PROXY=localhost,127.0.0.1,::1",
			"TMPDIR="+tmpRoot,
			"GOTMPDIR="+tmpRoot,
		)
		for i, step := range steps {
			cmd := exec.Command(step[0], step[1:]...)
			cmd.Dir = dirs[i]
			cmd.Env = env
			if out, err := cmd.CombinedOutput(); err != nil {
				buildErr = fmt.Errorf("build %v in %s: %v\n%s", step, dirs[i], err, out)
				return
			}
		}
	})
	if buildErr != nil {
		t.Fatalf("artifact build failed: %v", buildErr)
	}
}

func pluginRootDir(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return wd
}

type harness struct {
	t         *testing.T
	port      int
	cmd       *exec.Cmd
	traceFile string
	client    *http.Client
	clientKey string
	kimi      *fakeUpstream
	claudes   []*fakeUpstream
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// startHarness launches CPA with credit-manager + kimi-fit and N fake
// claude upstreams sharing the "high" alias with the fake kimi auth.
func startHarness(t *testing.T, numClaudes, kimiTriggerBytes, kimiLandingBytes, kimiQuantumBytes int) *harness {
	t.Helper()
	buildArtifacts(t)
	h := &harness{t: t, port: freePort(t), client: &http.Client{Timeout: 60 * time.Second}}
	h.kimi = newFakeUpstream(t)
	for i := 0; i < numClaudes; i++ {
		h.claudes = append(h.claudes, newFakeUpstream(t))
	}

	workDir := t.TempDir()
	h.traceFile = filepath.Join(workDir, "trace.jsonl")
	authDir := filepath.Join(workDir, "auths")
	cmData := filepath.Join(workDir, "cm-data")
	for _, d := range []string{authDir, cmData} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	kimiAuth := fmt.Sprintf(`{"type":"kimi","access_token":"fake-kimi","token_type":"Bearer","base_url":%q,"expired":"2099-01-01T00:00:00Z","priority":10}`, h.kimi.url())
	if err := os.WriteFile(filepath.Join(authDir, "kimi-fake.json"), []byte(kimiAuth), 0o600); err != nil {
		t.Fatal(err)
	}

	var apiKeysSection strings.Builder
	if len(h.claudes) > 0 {
		apiKeysSection.WriteString("api-keys:\n  claude:\n")
		for i, c := range h.claudes {
			fmt.Fprintf(&apiKeysSection, `      - name: "fake-claude-%d"
        base-url: %q
        models:
          - name: "claude-fake-%d"
            alias: %q
        keys:
          - api-key: "fake-claude-key-%d"
`, i, c.url(), i, modelAlias, i)
		}
	}

	cfg := fmt.Sprintf(`server:
  host: "127.0.0.1"
  port: %d
access:
  api-keys: ["ignored-by-exclusive-frontend-auth"]
management:
  secret-key: %q
  disable-control-panel: true
  disable-auto-update-panel: true
oauth:
  auth-dir: %q
  model-alias:
    kimi:
      - name: "kimi-k3.1-pro"
        alias: %q
        fork: true
%splugins:
  enabled: true
  dir: %q
  configs:
    credit-manager:
      enabled: true
      priority: 1
      data_dir: %q
      auth_quota:
        sync_interval: "0"
      scheduler:
        allow_provider_exclusions: true
    kimi-fit:
      enabled: true
      priority: 2
      nested_only: true
      trigger_bytes: %d
      landing_bytes: %d
      quantum_bytes: %d
      trace_file: %q
`, h.port, mgmtSecret, authDir, modelAlias, apiKeysSection.String(), filepath.Join(binDir, "plugins"), cmData, kimiTriggerBytes, kimiLandingBytes, kimiQuantumBytes, h.traceFile)

	cfgPath := filepath.Join(workDir, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte(cfg), 0o600); err != nil {
		t.Fatal(err)
	}

	h.cmd = exec.Command(filepath.Join(binDir, "cli-proxy-api"), "--config", cfgPath, "--local-model")
	h.cmd.Dir = workDir
	h.cmd.Env = append(os.Environ(),
		"NO_PROXY=localhost,127.0.0.1,::1",
		"HOME="+workDir,
	)
	logFile, err := os.Create(filepath.Join(workDir, "cpa.log"))
	if err != nil {
		t.Fatal(err)
	}
	h.cmd.Stdout, h.cmd.Stderr = logFile, logFile
	if err := h.cmd.Start(); err != nil {
		t.Fatalf("start cpa: %v", err)
	}
	t.Cleanup(func() {
		if h.cmd.Process != nil {
			_ = h.cmd.Process.Kill()
		}
		_ = h.cmd.Wait()
		if testing.Verbose() || t.Failed() {
			if raw, err := os.ReadFile(logFile.Name()); err == nil {
				t.Logf("cpa log tail:\n%s", tailBytes(raw, 4000))
			}
		}
	})
	h.waitReady(t)
	h.mintKey(t)
	return h
}

func tailBytes(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[len(b)-n:])
}

func (h *harness) waitReady(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	url := fmt.Sprintf("http://127.0.0.1:%d/v0/management/credit-manager/health", h.port)
	for time.Now().Before(deadline) {
		req, _ := http.NewRequest(http.MethodGet, url, nil)
		req.Header.Set("Authorization", "Bearer "+mgmtSecret)
		resp, err := h.client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		if h.cmd.ProcessState != nil && h.cmd.ProcessState.Exited() {
			t.Fatalf("cpa exited early")
		}
		time.Sleep(250 * time.Millisecond)
	}
	t.Fatal("cpa did not become ready in 60s")
}

func (h *harness) mintKey(t *testing.T) {
	t.Helper()
	url := fmt.Sprintf("http://127.0.0.1:%d/v0/management/credit-manager/keys", h.port)
	body := strings.NewReader(fmt.Sprintf(`{"label":"e2e","allowed_models":[%q]}`, modelAlias))
	req, _ := http.NewRequest(http.MethodPost, url, body)
	req.Header.Set("Authorization", "Bearer "+mgmtSecret)
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		t.Fatalf("mint key: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		t.Fatalf("mint key status %d: %s", resp.StatusCode, raw)
	}
	var out struct {
		Plaintext string `json:"plaintext"`
	}
	if err := json.Unmarshal(raw, &out); err != nil || out.Plaintext == "" {
		t.Fatalf("mint key did not return plaintext: %s", raw)
	}
	h.clientKey = out.Plaintext
}

func (h *harness) messages(t *testing.T, body []byte) (int, []byte) {
	t.Helper()
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/messages", h.port)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("x-api-key", h.clientKey)
	req.Header.Set("content-type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	resp, err := h.client.Do(req)
	if err != nil {
		t.Fatalf("messages: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, raw
}

func traceLines(t *testing.T, path string) []map[string]string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace: %v", err)
	}
	var out []map[string]string
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if line == "" {
			continue
		}
		var m map[string]string
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("bad trace line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func toolRefRequestBody(model string, maxTokens int) []byte {
	return []byte(fmt.Sprintf(`{"model":%q,"max_tokens":%d,"messages":[`, model, maxTokens) +
		`{"role":"user","content":[{"type":"text","text":"hi"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ToolSearch","input":{"query":"x"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` +
		`{"type":"tool_reference","tool_name":"WebSearch"}]}]},` +
		`{"role":"user","content":[{"type":"text","text":"latest question"}]}` +
		`]}`)
}

func oversizeRequestBody(model string, maxTokens, bigBytes int) []byte {
	big := strings.Repeat("x", bigBytes)
	return []byte(fmt.Sprintf(`{"model":%q,"max_tokens":%d,"messages":[`, model, maxTokens) +
		`{"role":"user","content":[{"type":"text","text":"` + big + `"}]}` +
		`]}`)
}

func withStream(body []byte) []byte {
	return append([]byte(`{"stream":true,`), body[1:]...)
}

func TestE2EKimiFlow(t *testing.T) {
	if testing.Short() {
		t.Skip("e2e")
	}
	h := startHarness(t, 2, 100_000, 75_000, 16_384)

	t.Run("fitting request reaches kimi normalized", func(t *testing.T) {
		status, raw := h.messages(t, toolRefRequestBody(modelAlias, 16))
		if status != http.StatusOK {
			t.Fatalf("status %d: %s", status, raw)
		}
		if h.kimi.requestCount() != 1 {
			t.Fatalf("kimi requests = %d, claude calls = %v", h.kimi.requestCount(), claudeCounts(h.claudes))
		}
		up := h.kimi.lastBody()
		if bytes.Contains(up, []byte(`"type":"tool_reference"`)) {
			t.Fatalf("tool_reference reached upstream: %s", tailBytes(up, 800))
		}
		if !bytes.Contains(up, []byte("WebSearch")) {
			t.Fatalf("tool name lost upstream")
		}
	})

	t.Run("oversize request fails over to claude with original body", func(t *testing.T) {
		big := 120_000 // > target 100000, in the protected latest message: untrimmable
		body := oversizeRequestBody(modelAlias, 16, big)
		status, raw := h.messages(t, body)
		if status != http.StatusOK {
			t.Fatalf("status %d: %s", status, raw)
		}
		if h.kimi.requestCount() != 1 {
			t.Fatalf("kimi got an extra request (count=%d): body should have failed over", h.kimi.requestCount())
		}
		var up []byte
		for _, c := range h.claudes {
			if c.requestCount() > 0 {
				up = c.lastBody()
			}
		}
		if up == nil {
			t.Fatal("no claude upstream received the request")
		}
		if !bytes.Contains(up, []byte(strings.Repeat("x", 60_000))) {
			t.Fatal("claude upstream did not receive the original large content")
		}
		if bytes.Contains(up, []byte("omitted")) {
			t.Fatalf("trim markers leaked into non-kimi body: %s", tailBytes(up, 400))
		}
	})

	t.Run("retry after claude failure never touches kimi", func(t *testing.T) {
		kimiBefore := h.kimi.requestCount()
		for _, c := range h.claudes {
			c.failOnce(http.StatusServiceUnavailable, `{"type":"error","error":{"type":"overloaded_error","message":"down"}}`)
		}
		body := oversizeRequestBody(modelAlias, 16, 120_000)
		status, _ := h.messages(t, body)
		if h.kimi.requestCount() != kimiBefore {
			t.Fatalf("kimi attempted during exhausted-candidate retry (count %d -> %d)", kimiBefore, h.kimi.requestCount())
		}
		if status == http.StatusOK {
			t.Fatal("expected failure after all claude candidates failed")
		}
	})

	t.Run("stream fitting request reaches kimi normalized", func(t *testing.T) {
		kimiBefore := h.kimi.requestCount()
		status, raw := h.messages(t, withStream(toolRefRequestBody(modelAlias, 16)))
		if status != http.StatusOK {
			t.Fatalf("stream status %d: %s", status, tailBytes(raw, 400))
		}
		if !bytes.Contains(raw, []byte("event:")) && !bytes.Contains(raw, []byte("message_start")) {
			t.Fatalf("not an SSE stream response: %s", tailBytes(raw, 200))
		}
		if h.kimi.requestCount() != kimiBefore+1 {
			t.Fatalf("stream request did not reach kimi (count %d -> %d)", kimiBefore, h.kimi.requestCount())
		}
		if bytes.Contains(h.kimi.lastBody(), []byte(`"type":"tool_reference"`)) {
			t.Fatal("tool_reference reached upstream on stream path")
		}
	})

	t.Run("trace order is before_auth,after_auth,complete", func(t *testing.T) {
		time.Sleep(300 * time.Millisecond)
		lines := traceLines(t, h.traceFile)
		var beforeIdx, afterIdx, completeIdx = -1, -1, -1
		// before/after lines carry the reservation header; completion lines do
		// not (no headers in RequestCompletion) — correlate them by request_id.
		nestedReqs := map[string]bool{}
		for i, l := range lines {
			if l["reservation"] != "" {
				nestedReqs[l["request_id"]] = true
			}
			switch {
			case strings.HasPrefix(l["phase"], "before_auth") && l["reservation"] != "" && beforeIdx < 0:
				beforeIdx = i
			case strings.HasPrefix(l["phase"], "after_auth") && l["reservation"] != "" && afterIdx < 0:
				afterIdx = i
			}
		}
		for i, l := range lines {
			if strings.HasPrefix(l["phase"], "complete:") && nestedReqs[l["request_id"]] && completeIdx < 0 {
				completeIdx = i
			}
		}
		if beforeIdx < 0 || afterIdx < 0 || completeIdx < 0 {
			t.Fatalf("missing traced phases: %+v", lines)
		}
		if !(beforeIdx < afterIdx && afterIdx < completeIdx) {
			t.Fatalf("callback order broken: %d,%d,%d", beforeIdx, afterIdx, completeIdx)
		}
	})
}

func TestE2EAllKimi413(t *testing.T) {
	if testing.Short() {
		t.Skip("e2e")
	}
	h := startHarness(t, 0, 100_000, 75_000, 16_384) // no claude candidates at all
	body := oversizeRequestBody(modelAlias, 16, 120_000)
	status, raw := h.messages(t, body)
	if status != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413: %s", status, tailBytes(raw, 400))
	}
	if !bytes.Contains(raw, []byte("request_too_large")) {
		t.Fatalf("missing request_too_large body: %s", tailBytes(raw, 400))
	}
	// NOTE: CPA wraps a plugin-terminated attempt as host_call_failed with the
	// status mapped to the Anthropic error type; the plugin's custom wording
	// is asserted at interceptor level (TestAfterAuthKimiNonFitting413Wording)
	// because it does not survive credit-manager's executor envelope.
	if h.kimi.requestCount() != 0 {
		t.Fatal("kimi upstream was contacted despite local 413")
	}
	// Kimi must NOT be cooled down by a local rejection: a small fitting
	// request right after must still reach it.
	status, raw = h.messages(t, toolRefRequestBody(modelAlias, 16))
	if status != http.StatusOK {
		t.Fatalf("post-413 fitting request status %d: %s", status, raw)
	}
	if h.kimi.requestCount() != 1 {
		t.Fatalf("kimi not selected after local 413 (cooldown suspicion), count=%d", h.kimi.requestCount())
	}
	// Stream variant of the same local rejection.
	status, raw = h.messages(t, withStream(body))
	if status != http.StatusRequestEntityTooLarge {
		t.Fatalf("stream status = %d, want 413: %s", status, tailBytes(raw, 400))
	}
	// CPA's stream terminate path does not propagate the plugin's response
	// body (it surfaces "host_call_failed: request terminated by plugin"),
	// so the wording assertion is non-stream only.
	if h.kimi.requestCount() != 1 {
		t.Fatalf("kimi contacted on stream 413 path, count=%d", h.kimi.requestCount())
	}
}

func claudeCounts(claudes []*fakeUpstream) []int {
	out := make([]int, len(claudes))
	for i, c := range claudes {
		out[i] = c.requestCount()
	}
	return out
}
