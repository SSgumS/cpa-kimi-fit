package plugin

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/router-for-me/kimi-fit-plugin/internal/planner"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

const (
	// reservationHeader is the trusted boundary created by credit-manager's
	// executor: it exists only inside nested executions, never from clients.
	reservationHeader = "X-Credit-Manager-Reservation"
	// excludeProvidersHeader carries the per-request exclusion set toward the
	// credit-manager scheduler; it is cleared again after auth selection.
	excludeProvidersHeader = "X-Credit-Manager-Exclude-Providers"
	// kimiExclusionValue lists every provider key CPA routes to KimiExecutor.
	kimiExclusionValue = "kimi,kimi-ai,kimi.ai,kimi.com"
)

type rpcRequestInterceptRequest struct {
	pluginapi.RequestInterceptRequest
	HostCallbackID string `json:"host_callback_id,omitempty"`
}

// interceptBeforeAuth plans edits without touching the body. A plan that
// cannot fit Kimi's byte budget emits the trusted provider-exclusion header so
// the scheduler picks another healthy candidate for this request only.
func interceptBeforeAuth(request []byte) ([]byte, error) {
	var req rpcRequestInterceptRequest
	if err := json.Unmarshal(request, &req); err != nil {
		return nil, err
	}
	reservationID := headerValueFold(req.Headers, reservationHeader)
	trace(cfg(), "before_auth", req, reservationID, "")

	if !isClaudeSource(req.SourceFormat) || len(req.Body) == 0 {
		return okEnvelope(pluginapi.RequestInterceptResponse{})
	}
	nested := reservationID != "" || metadataString(req.Metadata, "source") == "plugin_host_model_callback"
	if cfg().NestedOnly && !nested {
		return okEnvelope(pluginapi.RequestInterceptResponse{})
	}
	if len(req.Body) <= cfg().fastPathMaxBytes() && !planner.MayContainToolReference(req.Body) {
		return okEnvelope(pluginapi.RequestInterceptResponse{})
	}

	planKey, hasReservation := planKeyFor(req, reservationID)
	if planKey == "" {
		return okEnvelope(pluginapi.RequestInterceptResponse{})
	}
	plan, err := planner.BuildKimiRequestPlan(req.Body, cfg().planOptions())
	if err != nil {
		// Malformed body: fail open and let the provider answer.
		return okEnvelope(pluginapi.RequestInterceptResponse{})
	}
	if s := planStoreInstance(); s != nil {
		s.Put(planKey, req.RequestID, plan)
	}
	resp := pluginapi.RequestInterceptResponse{}
	if !plan.FitsKimi && hasReservation {
		resp.Headers = http.Header{excludeProvidersHeader: {kimiExclusionValue}}
	}
	return okEnvelope(resp)
}

// interceptAfterAuth applies the cached plan only when the selected credential
// is Kimi; every other provider receives the original body untouched.
func interceptAfterAuth(request []byte) ([]byte, error) {
	var req rpcRequestInterceptRequest
	if err := json.Unmarshal(request, &req); err != nil {
		return nil, err
	}
	reservationID := headerValueFold(req.Headers, reservationHeader)
	resp := pluginapi.RequestInterceptResponse{ClearHeaders: []string{excludeProvidersHeader}}

	if !isClaudeSource(req.SourceFormat) {
		trace(cfg(), "after_auth_skip_format", req, reservationID, "")
		return okEnvelope(resp)
	}
	authIndex := metadataString(req.Metadata, "selected_auth_index")
	if authIndex == "" {
		// Outer pass through a plugin executor has no selected auth: nothing
		// provider-specific can be decided here.
		trace(cfg(), "after_auth_no_auth", req, reservationID, "")
		return okEnvelope(resp)
	}
	provider, err := lookupAuthProvider(authIndex)
	if err != nil || !isKimiProvider(provider) {
		trace(cfg(), "after_auth_non_kimi", req, reservationID, provider)
		return okEnvelope(resp)
	}
	trace(cfg(), "after_auth_kimi", req, reservationID, provider)

	planKey, _ := planKeyFor(req, reservationID)
	if s := planStoreInstance(); s != nil && planKey != "" {
		if plan, ok := s.Get(planKey, req.Body); ok {
			if !plan.FitsKimi {
				return okEnvelope(localTooLargeResponse(resp, plan, cfg().TriggerBytes))
			}
			resp.BodyEdits = plan.Edits
			return okEnvelope(resp)
		}
	}
	// Cache miss or stale hash: re-plan inline rather than send a body we never
	// measured. Small, tool_reference-free bodies need no work.
	if len(req.Body) <= cfg().fastPathMaxBytes() && !planner.MayContainToolReference(req.Body) {
		return okEnvelope(resp)
	}
	plan, err := planner.BuildKimiRequestPlan(req.Body, cfg().planOptions())
	if err != nil {
		return okEnvelope(resp)
	}
	if !plan.FitsKimi {
		return okEnvelope(localTooLargeResponse(resp, plan, cfg().TriggerBytes))
	}
	resp.BodyEdits = plan.Edits
	return okEnvelope(resp)
}

// completeRequest drops the cached plan for this request; outer completions
// only ever delete outer keys, so a nested reservation plan survives.
func completeRequest(request []byte) ([]byte, error) {
	var completion pluginapi.RequestCompletion
	if err := json.Unmarshal(request, &completion); err != nil {
		return nil, err
	}
	trace(cfg(), "complete:"+string(completion.Outcome), rpcRequestInterceptRequest{
		RequestInterceptRequest: pluginapi.RequestInterceptRequest{
			RequestID: completion.RequestID,
			Metadata:  completion.Metadata,
		},
	}, "", "")
	if s := planStoreInstance(); s != nil {
		s.DeleteByRequestID(completion.RequestID)
	}
	return okEnvelope(map[string]any{})
}

func planKeyFor(req rpcRequestInterceptRequest, reservationID string) (key string, hasReservation bool) {
	if reservationID != "" {
		return "res:" + reservationID, true
	}
	if id := strings.TrimSpace(req.RequestID); id != "" {
		return "req:" + id, false
	}
	return "", false
}

func isClaudeSource(format string) bool {
	return strings.EqualFold(strings.TrimSpace(format), "claude")
}

// localTooLargeResponse terminates the attempt before any upstream call and
// without touching credential health: CPA maps the status to the Anthropic
// request_too_large error shape downstream. Above Kimi's hard limit the
// message says so; between trigger and hard limit it reports the budget.
func localTooLargeResponse(resp pluginapi.RequestInterceptResponse, plan planner.RequestPlan, triggerBytes int) pluginapi.RequestInterceptResponse {
	resp.Terminate = true
	resp.StatusCode = http.StatusRequestEntityTooLarge
	resp.ResponseHeaders = http.Header{"Content-Type": {"application/json"}}
	verb := "exceeds the request budget"
	limit := triggerBytes
	if plan.EstimatedBytes > kimiHardLimitBytes {
		verb = "exceeds Kimi's hard limit"
		limit = kimiHardLimitBytes
	}
	resp.ResponseBody = []byte(fmt.Sprintf(
		`{"type":"error","error":{"type":"request_too_large","message":"request body %s after permitted trimming (estimated %d bytes; limit %d)"}}`,
		verb, plan.EstimatedBytes, limit))
	return resp
}

// trace is a test-only hook enabled via the trace_file config knob; each
// callback appends one JSON line so Task 1 can assert real callback order.
var traceMu sync.Mutex

func trace(c Config, phase string, req rpcRequestInterceptRequest, reservationID, extra string) {
	if c.TraceFile == "" {
		return
	}
	line, _ := json.Marshal(map[string]string{
		"phase":        phase,
		"request_id":   req.RequestID,
		"source":       metadataString(req.Metadata, "source"),
		"reservation":  reservationID,
		"auth_index":   metadataString(req.Metadata, "selected_auth_index"),
		"exclude":      headerValueFold(req.Headers, excludeProvidersHeader),
		"extra":        extra,
		"body_bytes":   fmt.Sprintf("%d", len(req.Body)),
		"source_fmt":   req.SourceFormat,
	})
	traceMu.Lock()
	defer traceMu.Unlock()
	f, err := os.OpenFile(c.TraceFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}
