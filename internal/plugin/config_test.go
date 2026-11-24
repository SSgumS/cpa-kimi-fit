package plugin

import (
	"strings"
	"testing"
)

func TestParseConfigBudgetDefaults(t *testing.T) {
	cfg, err := ParseConfig(nil)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TriggerBytes != 1_966_080 || cfg.LandingBytes != 1_500_000 || cfg.QuantumBytes != 262_144 {
		t.Fatalf("budget defaults = %d/%d/%d", cfg.TriggerBytes, cfg.LandingBytes, cfg.QuantumBytes)
	}
	if cfg.fastPathMaxBytes() != 1_474_560 {
		t.Fatalf("auto fast path = %d, want 1474560", cfg.fastPathMaxBytes())
	}
}

func TestParseConfigBudgetCustomRoundTrip(t *testing.T) {
	cfg, err := ParseConfig([]byte("trigger_bytes: 100000\nlanding_bytes: 75000\nquantum_bytes: 16384\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TriggerBytes != 100000 || cfg.LandingBytes != 75000 || cfg.QuantumBytes != 16384 {
		t.Fatalf("parsed = %d/%d/%d", cfg.TriggerBytes, cfg.LandingBytes, cfg.QuantumBytes)
	}
	if cfg.fastPathMaxBytes() != 75000 {
		t.Fatalf("auto fast path = %d, want 75000", cfg.fastPathMaxBytes())
	}
}

func TestParseConfigRejectsDeprecatedTargetBytes(t *testing.T) {
	for _, raw := range []string{
		"target_bytes: 1900000\n",
		"target_bytes: 1900000\ntrigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\n",
	} {
		if _, err := ParseConfig([]byte(raw)); err == nil || !strings.Contains(err.Error(), "target_bytes") {
			t.Fatalf("deprecated key not rejected with guidance: %q -> %v", raw, err)
		}
	}
}

func TestParseConfigBudgetValidation(t *testing.T) {
	maxTrigger := kimiHardLimitBytes - cpaMarginBytes
	cases := []struct {
		name string
		yaml string
	}{
		{"landing equals trigger", "trigger_bytes: 1000\nlanding_bytes: 1000\nquantum_bytes: 256\n"},
		{"landing above trigger", "trigger_bytes: 1000\nlanding_bytes: 2000\nquantum_bytes: 256\n"},
		{"zero quantum", "trigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 0\n"},
		{"fast path at trigger", "trigger_bytes: 1000\nlanding_bytes: 750\nquantum_bytes: 256\nfast_path_max_bytes: 1000\n"},
		{"trigger one past margin", "trigger_bytes: " + itoa(maxTrigger+1) + "\nlanding_bytes: 1000\nquantum_bytes: 256\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseConfig([]byte(tc.yaml)); err == nil {
				t.Fatalf("invalid budget accepted: %s", tc.yaml)
			}
		})
	}
	// Exact margin boundary is valid.
	if _, err := ParseConfig([]byte("trigger_bytes: " + itoa(maxTrigger) + "\nlanding_bytes: 1000\nquantum_bytes: 256\n")); err != nil {
		t.Fatalf("exact margin boundary rejected: %v", err)
	}
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

func TestParseConfigExplicitFastPath(t *testing.T) {
	cfg, err := ParseConfig([]byte("fast_path_max_bytes: 5000\n"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.fastPathMaxBytes() != 5000 {
		t.Fatalf("explicit fast path = %d, want 5000", cfg.fastPathMaxBytes())
	}
}

func TestParseConfigLengthOptimizationDefaultOff(t *testing.T) {
	cfg, err := ParseConfig(nil)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LengthOptimizationEnabled {
		t.Fatal("length_optimization_enabled must default to false")
	}
	if !cfg.planOptions().DisableLengthTrim {
		t.Fatal("plan options must carry DisableLengthTrim when optimization is off")
	}
	cfg, err = ParseConfig([]byte("length_optimization_enabled: true\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.LengthOptimizationEnabled {
		t.Fatal("length_optimization_enabled: true not parsed")
	}
	if cfg.planOptions().DisableLengthTrim {
		t.Fatal("plan options must not disable trimming when optimization is on")
	}
}
