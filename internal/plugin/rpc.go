package plugin

import (
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

const (
	pluginName    = "kimi-fit"
	pluginVersion = "0.1.2"
)

type registration struct {
	SchemaVersion uint32                 `json:"schema_version"`
	Metadata      pluginapi.Metadata     `json:"metadata"`
	Capabilities  registrationCapability `json:"capabilities"`
}

type registrationCapability struct {
	RequestInterceptor     bool `json:"request_interceptor"`
	RequestLifecyclePlugin bool `json:"request_lifecycle_plugin"`
}

func pluginRegistration(schemaVersion uint32) registration {
	return registration{
		SchemaVersion: schemaVersion,
		Metadata: pluginapi.Metadata{
			Name:             pluginName,
			Version:          pluginVersion,
			Author:           "router-for-me",
			GitHubRepository: "https://github.com/router-for-me/kimi-fit-plugin",
			ConfigFields:     pluginConfigFields(),
		},
		Capabilities: registrationCapability{
			RequestInterceptor:     true,
			RequestLifecyclePlugin: true,
		},
	}
}

// Runtime state, all reset on (re)configure.
var (
	activeConfig atomic.Pointer[Config]
	stateMu      sync.Mutex
	store        *planStore
	providerC    *authProviderCache
)

func providerCache() *authProviderCache {
	stateMu.Lock()
	defer stateMu.Unlock()
	return providerC
}

func planStoreInstance() *planStore {
	stateMu.Lock()
	defer stateMu.Unlock()
	return store
}

func cfg() Config {
	if c := activeConfig.Load(); c != nil {
		return *c
	}
	return DefaultConfig()
}

func configure(req lifecycleRequest) error {
	c, err := ParseConfig(req.ConfigYAML)
	if err != nil {
		return err
	}
	ttl, _ := c.planCacheTTL()
	authTTL, _ := c.authProviderCacheTTL()
	stateMu.Lock()
	defer stateMu.Unlock()
	activeConfig.Store(&c)
	store = newPlanStore(c.PlanCacheMaxEntries, ttl, timeNow)
	providerC = newAuthProviderCache(authTTL)
	return nil
}

// Shutdown releases runtime state on plugin.quiesce/shutdown.
func Shutdown() {
	stateMu.Lock()
	defer stateMu.Unlock()
	store = nil
	providerC = nil
}

// timeNow is swapped by tests to freeze plan-store TTL expiry.
var timeNow = time.Now

func HandleMethod(method string, request []byte) ([]byte, error) {
	switch method {
	case pluginabi.MethodPluginRegister, pluginabi.MethodPluginReconfigure:
		req, err := decodeLifecycle(request)
		if err != nil {
			return nil, err
		}
		if err := configure(req); err != nil {
			return nil, err
		}
		return okEnvelope(pluginRegistration(negotiateRPCSchema(req.SchemaVersion)))
	case pluginabi.MethodPluginQuiesce:
		Shutdown()
		return okEnvelope(map[string]any{})
	case pluginabi.MethodRequestInterceptBefore:
		return interceptBeforeAuth(request)
	case pluginabi.MethodRequestInterceptAfter:
		return interceptAfterAuth(request)
	case pluginabi.MethodRequestComplete:
		return completeRequest(request)
	default:
		return errorEnvelope("unknown_method", "unknown method: "+method), nil
	}
}

func okEnvelope(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return json.Marshal(envelope{OK: true, Result: raw})
}

func errorEnvelope(code, message string) []byte {
	raw, _ := json.Marshal(envelope{OK: false, Error: &envelopeError{Code: code, Message: message}})
	return raw
}

// ErrorEnvelope is used by the C ABI entrypoint for top-level failures.
func ErrorEnvelope(code, message string) []byte {
	return errorEnvelope(code, message)
}

func headerValueFold(headers map[string][]string, name string) string {
	for k, values := range headers {
		if !strings.EqualFold(k, name) {
			continue
		}
		for _, v := range values {
			if v = strings.TrimSpace(v); v != "" {
				return v
			}
		}
	}
	return ""
}

func metadataString(md map[string]any, key string) string {
	if md == nil {
		return ""
	}
	if s, ok := md[key].(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}
