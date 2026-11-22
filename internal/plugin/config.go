package plugin

import (
	"fmt"
	"time"

	"github.com/router-for-me/kimi-compat-plugin/internal/planner"

	"github.com/router-for-me/CLIProxyAPI/v8/sdk/pluginapi"
	"gopkg.in/yaml.v3"
)

// Config is the plugin YAML delivered via plugin.register/plugin.reconfigure.
type Config struct {
	TriggerBytes         int    `yaml:"trigger_bytes"`
	LandingBytes         int    `yaml:"landing_bytes"`
	QuantumBytes         int    `yaml:"quantum_bytes"`
	FastPathMaxBytes     int    `yaml:"fast_path_max_bytes"`
	ProtectedMessages    int    `yaml:"protected_recent_messages"`
	ProtectedImages      int    `yaml:"protected_recent_images"`
	LargeToolResultBytes int    `yaml:"large_tool_result_bytes"`
	ToolResultHeadBytes  int    `yaml:"tool_result_head_bytes"`
	ToolResultTailBytes  int    `yaml:"tool_result_tail_bytes"`
	PlanCacheMaxEntries  int    `yaml:"plan_cache_max_entries"`
	PlanCacheTTL         string `yaml:"plan_cache_ttl"`
	// NestedOnly skips before-auth planning on the outer pass when running
	// behind a plugin executor like credit-manager (the nested pass carries
	// the reservation header and does the real planning).
	NestedOnly           bool   `yaml:"nested_only"`
	AuthProviderCacheTTL string `yaml:"auth_provider_cache_ttl"`
	// TraceFile is a test hook: when set, each interceptor callback appends a
	// JSON line recording phase and metadata for callback-order assertions.
	TraceFile string `yaml:"trace_file,omitempty"`
}

const kimiHardLimitBytes = 2_097_152

// cpaMarginBytes reserves headroom under Kimi's hard limit for additions the
// executor may make after the plugin measures the body (cache markers,
// thinking replay); the trigger sits below it.
const cpaMarginBytes = 128 * 1024

func DefaultConfig() Config {
	defaults := planner.DefaultPlanOptions()
	return Config{
		TriggerBytes:         defaults.TriggerBytes,
		LandingBytes:         defaults.LandingBytes,
		QuantumBytes:         defaults.QuantumBytes,
		FastPathMaxBytes:     0, // auto: trigger*3/4
		ProtectedMessages:    defaults.ProtectedMessages,
		ProtectedImages:      defaults.ProtectedImages,
		LargeToolResultBytes: defaults.LargeToolResultBytes,
		ToolResultHeadBytes:  defaults.ToolResultHeadBytes,
		ToolResultTailBytes:  defaults.ToolResultTailBytes,
		PlanCacheMaxEntries:  128,
		PlanCacheTTL:         "10m",
		AuthProviderCacheTTL: "60s",
	}
}

func ParseConfig(raw []byte) (Config, error) {
	cfg := DefaultConfig()
	if len(raw) > 0 {
		var keys map[string]any
		if err := yaml.Unmarshal(raw, &keys); err != nil {
			return Config{}, fmt.Errorf("parse kimi-compat config: %w", err)
		}
		if _, deprecated := keys["target_bytes"]; deprecated {
			return Config{}, fmt.Errorf("target_bytes is removed: set trigger_bytes (<= %d), landing_bytes (< trigger), and quantum_bytes instead",
				kimiHardLimitBytes-cpaMarginBytes)
		}
		if err := yaml.Unmarshal(raw, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse kimi-compat config: %w", err)
		}
	}
	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if c.TriggerBytes <= 0 {
		return fmt.Errorf("trigger_bytes must be > 0")
	}
	if c.LandingBytes <= 0 {
		return fmt.Errorf("landing_bytes must be > 0")
	}
	if c.TriggerBytes <= c.LandingBytes {
		return fmt.Errorf("trigger_bytes (%d) must be greater than landing_bytes (%d)", c.TriggerBytes, c.LandingBytes)
	}
	if c.QuantumBytes <= 0 {
		return fmt.Errorf("quantum_bytes must be > 0")
	}
	if c.TriggerBytes > kimiHardLimitBytes-cpaMarginBytes {
		return fmt.Errorf("trigger_bytes must be <= %d (hard limit %d minus %d executor margin)", kimiHardLimitBytes-cpaMarginBytes, kimiHardLimitBytes, cpaMarginBytes)
	}
	if c.FastPathMaxBytes < 0 || c.FastPathMaxBytes >= c.TriggerBytes {
		return fmt.Errorf("fast_path_max_bytes must be in [0, trigger_bytes)")
	}
	if c.ProtectedMessages < 1 {
		return fmt.Errorf("protected_recent_messages must be >= 1")
	}
	if c.ProtectedImages < 0 {
		return fmt.Errorf("protected_recent_images must be >= 0")
	}
	if c.LargeToolResultBytes <= 0 {
		return fmt.Errorf("large_tool_result_bytes must be > 0")
	}
	if c.ToolResultHeadBytes < 0 || c.ToolResultTailBytes < 0 {
		return fmt.Errorf("tool_result head/tail bytes must be >= 0")
	}
	if c.ToolResultHeadBytes+c.ToolResultTailBytes >= c.LargeToolResultBytes {
		return fmt.Errorf("tool_result_head_bytes + tool_result_tail_bytes must be < large_tool_result_bytes")
	}
	if c.PlanCacheMaxEntries < 1 || c.PlanCacheMaxEntries > 4096 {
		return fmt.Errorf("plan_cache_max_entries must be in [1, 4096]")
	}
	if _, err := c.planCacheTTL(); err != nil {
		return err
	}
	if _, err := c.authProviderCacheTTL(); err != nil {
		return err
	}
	return nil
}

func (c Config) planCacheTTL() (time.Duration, error) {
	d, err := time.ParseDuration(c.PlanCacheTTL)
	if err != nil || d <= 0 || d > time.Hour {
		return 0, fmt.Errorf("plan_cache_ttl must parse as a duration in (0, 1h], got %q", c.PlanCacheTTL)
	}
	return d, nil
}

func (c Config) authProviderCacheTTL() (time.Duration, error) {
	d, err := time.ParseDuration(c.AuthProviderCacheTTL)
	if err != nil || d < 0 || d > 10*time.Minute {
		return 0, fmt.Errorf("auth_provider_cache_ttl must parse as a duration in [0, 10m], got %q", c.AuthProviderCacheTTL)
	}
	return d, nil
}

func (c Config) fastPathMaxBytes() int {
	if c.FastPathMaxBytes > 0 {
		return c.FastPathMaxBytes
	}
	return c.TriggerBytes * 3 / 4
}

func (c Config) planOptions() planner.PlanOptions {
	return planner.PlanOptions{
		TriggerBytes:         c.TriggerBytes,
		LandingBytes:         c.LandingBytes,
		QuantumBytes:         c.QuantumBytes,
		ProtectedMessages:    c.ProtectedMessages,
		ProtectedImages:      c.ProtectedImages,
		LargeToolResultBytes: c.LargeToolResultBytes,
		ToolResultHeadBytes:  c.ToolResultHeadBytes,
		ToolResultTailBytes:  c.ToolResultTailBytes,
	}
}

func pluginConfigFields() []pluginapi.ConfigField {
	return []pluginapi.ConfigField{
		{Name: "trigger_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 1_966_080, Description: "Raw body size above which duplicate images are replaced and trimming applies; final estimate must be at or below this (Kimi hard limit 2097152 minus 128 KiB executor margin)."},
		{Name: "landing_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 1_500_000, Description: "Size the quantized trim budget aims toward."},
		{Name: "quantum_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 262_144, Description: "Required savings round up to this multiple, spacing cache-invalidating trims into batches."},
		{Name: "fast_path_max_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 0, Description: "Bodies at or below this size skip JSON parsing when no tool_reference exists (0 = 3/4 of trigger_bytes)."},
		{Name: "protected_recent_messages", Type: pluginapi.ConfigFieldTypeInteger, Default: 24, Description: "Trailing messages never trimmed."},
		{Name: "protected_recent_images", Type: pluginapi.ConfigFieldTypeInteger, Default: 2, Description: "Newest images never removed."},
		{Name: "large_tool_result_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 16384, Description: "Old tool_result texts larger than this may be head/tail trimmed."},
		{Name: "tool_result_head_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 4096, Description: "UTF-8 bytes kept at the start of a trimmed tool_result."},
		{Name: "tool_result_tail_bytes", Type: pluginapi.ConfigFieldTypeInteger, Default: 4096, Description: "UTF-8 bytes kept at the end of a trimmed tool_result."},
		{Name: "plan_cache_max_entries", Type: pluginapi.ConfigFieldTypeInteger, Default: 128, Description: "Bounded plan cache entries."},
		{Name: "plan_cache_ttl", Type: pluginapi.ConfigFieldTypeString, Default: "10m", Description: "Plan cache entry lifetime."},
		{Name: "nested_only", Type: pluginapi.ConfigFieldTypeBoolean, Default: false, Description: "Skip outer-pass planning when running behind a nested plugin executor."},
		{Name: "auth_provider_cache_ttl", Type: pluginapi.ConfigFieldTypeString, Default: "60s", Description: "Selected-auth provider lookup cache lifetime (0s disables)."},
	}
}
