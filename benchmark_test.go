package main_test

// Benchmarks for the planner hot paths using synthetic request fixtures.
// Run: go test -run '^$' -bench . -benchmem ./...

import (
	"crypto/sha256"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/router-for-me/kimi-fit-plugin/internal/planner"
)

// deterministicImage returns a deterministic pseudo-random image payload of
// rawBytes, derived from seed so duplicate seeds produce duplicate images.
func deterministicImage(seed string, rawBytes int) string {
	var b strings.Builder
	b.Grow(rawBytes)
	counter := 0
	for b.Len() < rawBytes {
		sum := sha256.Sum256([]byte(seed + string(rune('a'+counter%26))))
		b.Write(sum[:])
		counter++
	}
	return base64.StdEncoding.EncodeToString([]byte(b.String()[:rawBytes]))
}

func benchBody(toolRefs, images, toolResults, filler int) []byte {
	var b strings.Builder
	b.WriteString(`{"model":"k3.1-pro","messages":[`)
	for i := 0; i < filler/2; i++ {
		b.WriteString(`{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("w", 200) + `"}]},`)
	}
	// Media interleaved in the middle: old enough to fall outside both the
	// trailing window and the active-turn fence, and mixed so the age prefix
	// hits images and tool_results alike. Images repeat seeds so duplicates
	// exist.
	for i := 0; i < toolRefs; i++ {
		b.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t` + string(rune('a'+i%26)) + `","content":[{"type":"tool_reference","tool_name":"WebSearch"}]}]},`)
	}
	seeds := []string{"img-A", "img-A", "img-B", "img-C", "img-D", "img-E"}
	media := max(images, toolResults)
	for i := 0; i < media; i++ {
		if i < images {
			b.WriteString(`{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + deterministicImage(seeds[i%len(seeds)], 256*1024) + `"}}]},`)
		}
		if i < toolResults {
			b.WriteString(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"r` + string(rune('a'+i%26)) + `","content":"` + strings.Repeat("z", 40*1024) + `"}]},`)
		}
	}
	for i := 0; i < filler-filler/2; i++ {
		b.WriteString(`{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("w", 200) + `"}]},`)
	}
	b.WriteString(`{"role":"user","content":[{"type":"text","text":"latest"}]}]}`)
	return []byte(b.String())
}

func BenchmarkPlanSmallBody(b *testing.B) {
	body := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := planner.BuildKimiRequestPlan(body, planner.DefaultPlanOptions()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPlanNormalizeOnly(b *testing.B) {
	body := benchBody(4, 0, 0, 40)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := planner.BuildKimiRequestPlan(body, planner.DefaultPlanOptions()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPlanTrimHeavy(b *testing.B) {
	body := benchBody(2, 6, 5, 60)
	b.Logf("body=%d bytes", len(body))
	// The heavy benchmark must exercise the real trim path: dedup, image
	// omission, and tool_result shortening all fire under default budgets.
	plan, err := planner.BuildKimiRequestPlan(body, planner.DefaultPlanOptions())
	if err != nil {
		b.Fatal(err)
	}
	if !plan.FitsKimi {
		b.Fatalf("heavy fixture does not fit: %d bytes estimated, stats %+v", plan.EstimatedBytes, plan.Stats)
	}
	if plan.Stats.DuplicateImages == 0 || plan.Stats.OldImages == 0 || plan.Stats.TrimmedToolResults == 0 {
		b.Fatalf("trim path not exercised: %+v", plan.Stats)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := planner.BuildKimiRequestPlan(body, planner.DefaultPlanOptions()); err != nil {
			b.Fatal(err)
		}
	}
}
