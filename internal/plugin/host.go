package plugin

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginabi"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
)

// HostCall is the CGO host RPC. main sets this during init; tests stub it.
var HostCall func(method string, payload []byte) (response []byte, code int, err error)

type envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *envelopeError  `json:"error,omitempty"`
}

type envelopeError struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	HTTPStatus int    `json:"http_status,omitempty"`
}

func callHost(method string, payload any) (json.RawMessage, error) {
	if HostCall == nil {
		return nil, fmt.Errorf("host callback %s: not configured", method)
	}
	rawPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal host callback %s: %w", method, err)
	}
	rawResponse, callCode, err := HostCall(method, rawPayload)
	if err != nil {
		return nil, err
	}
	if len(rawResponse) == 0 {
		return nil, fmt.Errorf("host callback %s returned no response, code=%d", method, callCode)
	}
	var env envelope
	if err := json.Unmarshal(rawResponse, &env); err != nil {
		return nil, fmt.Errorf("decode host envelope %s: %w", method, err)
	}
	if !env.OK {
		if env.Error != nil {
			return nil, fmt.Errorf("%s: %s", env.Error.Code, env.Error.Message)
		}
		return nil, fmt.Errorf("host callback %s failed", method)
	}
	return append(json.RawMessage(nil), env.Result...), nil
}

var kimiProviderAliases = map[string]struct{}{
	"kimi":     {},
	"kimi-ai":  {},
	"kimi.ai":  {},
	"kimi.com": {},
}

func isKimiProvider(provider string) bool {
	_, ok := kimiProviderAliases[strings.ToLower(strings.TrimSpace(provider))]
	return ok
}

// firstNonEmptyProvider picks the authoritative provider field of an auth
// entry.
func authEntryProvider(auth pluginapi.HostAuthFileEntry) string {
	if p := strings.TrimSpace(auth.Provider); p != "" {
		return p
	}
	return strings.TrimSpace(auth.Type)
}

type authCacheEntry struct {
	provider string
	expires  time.Time
}

type authProviderCache struct {
	mu      sync.Mutex
	entries map[string]authCacheEntry
	ttl     time.Duration
	now     func() time.Time
}

func newAuthProviderCache(ttl time.Duration) *authProviderCache {
	return &authProviderCache{entries: map[string]authCacheEntry{}, ttl: ttl, now: time.Now}
}

func (c *authProviderCache) get(authIndex string) (string, bool) {
	if c == nil || c.ttl <= 0 {
		return "", false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[authIndex]
	if !ok || c.now().After(e.expires) {
		return "", false
	}
	return e.provider, true
}

func (c *authProviderCache) put(authIndex, provider string) {
	if c == nil || c.ttl <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.entries) >= 256 {
		// Bounded: drop one arbitrary expired-or-first entry when full.
		for k, e := range c.entries {
			if c.now().After(e.expires) {
				delete(c.entries, k)
				break
			}
		}
		if len(c.entries) >= 256 {
			for k := range c.entries {
				delete(c.entries, k)
				break
			}
		}
	}
	c.entries[authIndex] = authCacheEntry{provider: provider, expires: c.now().Add(c.ttl)}
}

func (c *authProviderCache) flush() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.entries = map[string]authCacheEntry{}
	c.mu.Unlock()
}

// lookupAuthProvider resolves the selected credential's provider through the
// host's runtime auth entry, with a short TTL cache.
func lookupAuthProvider(authIndex string) (string, error) {
	authIndex = strings.TrimSpace(authIndex)
	if authIndex == "" {
		return "", errors.New("empty auth index")
	}
	if cache := providerCache(); cache != nil {
		if provider, ok := cache.get(authIndex); ok {
			return provider, nil
		}
	}
	raw, err := callHost(pluginabi.MethodHostAuthGetRuntime, pluginapi.HostAuthGetRequest{AuthIndex: authIndex})
	if err != nil {
		return "", err
	}
	var resp pluginapi.HostAuthGetRuntimeResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("decode auth runtime: %w", err)
	}
	provider := authEntryProvider(resp.Auth)
	if cache := providerCache(); cache != nil {
		cache.put(authIndex, provider)
	}
	return provider, nil
}
